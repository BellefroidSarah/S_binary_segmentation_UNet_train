from cytomine.models import ImageInstanceCollection, AnnotationCollection, TermCollection, Job, AttachedFile
from collections import defaultdict
from cytomine import CytomineJob
from pathlib import Path
from shapely import wkt
from unet import UNET
from PIL import Image

import numpy as np

import dataset
import joblib
import utils
import torch
import sys
import cv2
import os


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(status=Job.RUNNING, progress=0, status_comment="Initialization...")

        # Fetching the name of the term
        terms_collection = TermCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        term_name = ""
        for term in terms_collection:
            if term.id == cj.parameters.cytomine_term_id:
                term_name = term.name

        # Creating directories
        DIR = str(Path.home())

        dataset_path = os.path.join(DIR, "dataset")
        images_path = os.path.join(dataset_path, "images")
        mask_path = os.path.join(dataset_path, term_name)
        model_path = os.path.join(DIR, "models", str(cj.parameters.cytomine_id_project))
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # Fetching annotations
        cj.job.update(progress=5, status_comment="Downloading images and annotations...")
        annotations = AnnotationCollection()
        annotations.project = cj.parameters.cytomine_id_project
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.term = cj.parameters.cytomine_term_id
        annotations.fetch()

        annot_per_image = defaultdict(list)
        for annot in annotations:
            annot_per_image[annot.image].append(wkt.loads(annot.location))

        # Fetching images
        images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)

        # Downloading images and creating masks
        filepaths = list()
        for img in images:
            name = img.originalFilename[:-4] + ".jpg"
            filepath = os.path.join(images_path, name)
            filepaths.append(filepath)

            img.download(filepath, override=True)

            # Fetching image size
            pil_image = Image.open(filepath)
            shape = (pil_image.height, pil_image.width)

            # Creating masks
            if annot_per_image[img.id]:
                mask = np.zeros((shape[0], shape[1]))
                for g in annot_per_image[img.id]:
                    poly_points = []
                    for x, y in g.exterior.coords[:]:
                        poly_points.append([x, y])
                    poly_points = np.array(poly_points)
                    poly_points = np.int32(poly_points)
                    cv2.fillPoly(mask, [poly_points], color=255)
                cv2.imwrite(os.path.join(mask_path, name), np.flipud(mask))

        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = 'cuda:0'
        DEVICE = torch.device(device_name)

        # Starting training with K-fold cross validation
        # Best model of all folds
        best_model = UNET(3, 2)
        fold_loss = np.inf

        cj.job.update(progress=10, status_comment="Training...")
        for fold in range(cj.parameters.n_k_folds):
            # Creating datasets
            training_set = dataset.DatasetKfold(images_path,
                                                mask_path,
                                                fold,
                                                dataset="train",
                                                folds=cj.parameters.n_k_folds)

            validation_set = dataset.DatasetKfold(images_path,
                                                  mask_path,
                                                  fold,
                                                  dataset="validate",
                                                  folds=cj.parameters.n_k_folds)

            training_loader = torch.utils.data.DataLoader(training_set,
                                                          batch_size=cj.parameters.batch_size,
                                                          shuffle=True,
                                                          num_workers=0)

            validation_loader = torch.utils.data.DataLoader(validation_set,
                                                            batch_size=cj.parameters.batch_size,
                                                            shuffle=True,
                                                            num_workers=0)

            # Model
            model = UNET(3, 2)
            model.to(DEVICE)

            # Selecting loss function
            criterion = utils.select_loss_function(cj)

            # Optimiser
            optimiser = torch.optim.Adam(model.parameters(), lr=cj.parameters.learning_rate,
                                         weight_decay=cj.parameters.weight_decay)

            cj.job.update(progress=10 + 10 * (fold + 1), status_comment="Starting fold " + str(fold + 1) + "...")
            for epoch in range(cj.parameters.n_epochs):
                # Train + Validate
                _ = utils.train_model(model, training_loader, DEVICE, criterion, optimiser)
                v_loss = utils.validate_model(model, validation_loader, DEVICE, criterion)

                # Updating the overall best model
                if v_loss < fold_loss:
                    best_model = model
                    fold_loss = v_loss

        # Uploading model to cytomine
        cj.job.update(progress=95, status_comment="Finished training")
        model_filepath = os.path.join(model_path, "model.pth")
        torch.save(best_model.state_dict(), model_filepath)

        AttachedFile(cj.job,
                     domainIdent=cj.job.id,
                     filename=model_filepath,
                     domainClassName='be.cytomine.processing.Job'
                     ).upload()

        # Importing parameters needed for prediction to cytomine
        parameters_to_save = {'cytomine_term_id': cj.parameters.cytomine_term_id, 'cytomine_term_name': term_name}
        parameters_file = joblib.dump(parameters_to_save, os.path.join(model_path, "parameters"), compress=0)[0]

        AttachedFile(cj.job,
                     domainIdent=cj.job.id,
                     filename=parameters_file,
                     domainClassName='be.cytomine.processing.Job'
                     ).upload()

        cj.job.update(status=Job.TERMINATED, progress=100, statusComment='Job completed.')


if __name__ == '__main__':
    main(sys.argv[1:])
