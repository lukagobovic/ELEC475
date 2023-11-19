import torch
import os
import cv2
import argparse
from model import YodaClassifier  # Replace with your YODA Classifier code
from KittiDataset import KittiDataset
from KittiAnchors import Anchors
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt

def visualize_detected_and_ground_truth(image, detected_cars, ground_truth):
    image_copy = image.copy()
    for box in detected_cars:
        cv2.rectangle(image_copy, (box[0][1], box[0][0]), (box[1][1], box[1][0]), color=(0, 255, 0), thickness=1)
    for box in ground_truth:
        cv2.rectangle(image_copy, (box[0][1], box[0][0]), (box[1][1], box[1][0]), color=(0, 0, 255), thickness=1)

    cv2.imshow('Detected and Ground Truth', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def your_preprocessing_function(roi):
    # Resize to the desired input size
    roi = cv2.resize(roi, (150, 150))
    # Convert to PyTorch tensor
    roi_tensor = transforms.ToTensor()(roi)

    roi_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(roi_tensor)
    return roi_tensor

def calculate_mean_iou(all_detected_cars, ground_truth_cars_list, detected_indices):
    total_iou = 0
    total_detected_cars = len(detected_indices)

    for idx in detected_indices:
        detected_cars = all_detected_cars[idx]
        ground_truth_cars = ground_truth_cars_list[idx]

        for detected_car in detected_cars:
            max_iou = 0

            for ground_truth_car in ground_truth_cars:
                # Calculate IoU for the pair (detected_car, ground_truth_car)
                iou = Anchors().calc_IoU(detected_car, ground_truth_car)
                max_iou = max(max_iou, iou)

            total_iou += max_iou

        # Check for any missed ground truth cars
        for ground_truth_car in ground_truth_cars:
            if all(detected_car is not None for detected_car in detected_cars):
                continue

            # Calculate IoU for the pair (None, ground_truth_car)
            iou = Anchors().calc_IoU(None, ground_truth_car)
            total_iou += iou

    mean_iou = total_iou / total_detected_cars if total_detected_cars > 0 else 0
    return mean_iou

def save_bounding_boxes(image, boxes, output_dir, idx):
    image_with_boxes = image.copy()
    for k, box in enumerate(boxes):
        pt1 = (box[0][1], box[0][0])
        pt2 = (box[1][1], box[1][0])
        cv2.rectangle(image_with_boxes, pt1, pt2, color=(0, 255, 255), thickness=1)

    filename = f"{idx}_all_boxes.png"
    cv2.imwrite(os.path.join(output_dir, filename), image_with_boxes)

def main():
    print('running YODA test...')

    output_dir = 'output'  # Change this as needed

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')

    args = argParser.parse_args()

    input_dir = None
    if args.i is not None:
        input_dir = args.i

    output_dir = None
    if args.o is not None:
        output_dir = args.o

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(input_dir, training=False)
    num_classes = 2
    model = YodaClassifier(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.load_state_dict(torch.load('yoda_classifier_with_early_stopping_LR00001B96.pth'))

    i = 0
    ground_truth_cars_list = []  
    all_detected_cars = []  # Initialize a list to store detected cars for all images
    detected_indices = []
    for item in enumerate(dataset):
        # if i == 50:
        #     break
        idx = item[0]
        image = item[1][0]
        label = item[1][1]

        # print(i, idx, label)

        idx = dataset.class_label['Car']
        ground_truth_cars = dataset.strip_ROIs(class_ID=idx, label_list=label)
        ground_truth_cars_list.append(ground_truth_cars)

        # Display ground truth cars
        image_ground_truth = image.copy()
        for box in ground_truth_cars:
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(image_ground_truth, pt1, pt2, color=(0, 0, 255), thickness=1)

        # Show the images
        # cv2.imshow('Ground Truth Cars', image_ground_truth)
        # cv2.waitKey(0)


        anchor_centers = Anchors().calc_anchor_centers(image.shape, Anchors().grid)
        ROIs, boxes = Anchors().get_anchor_ROIs(image, anchor_centers, Anchors().shapes)
        # save_bounding_boxes(image.copy(), boxes, output_dir, idx)  # Save bounding box images

        # Save bounding box images
        save_bounding_boxes(image.copy(), boxes, output_dir, idx)

        # Display all ROIs
        image_all_rois = image.copy()
        for box in boxes:
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(image_all_rois, pt1, pt2, color=(255, 0, 0), thickness=1)

        # cv2.imshow('All ROIs', image_all_rois)
        # cv2.waitKey(0)

        roi_batch = torch.stack([your_preprocessing_function(roi) for roi in ROIs])
        model.eval()
        with torch.no_grad():
            yoda_output = model(roi_batch.to(device))

        detected_cars = []
        for k in range(len(boxes)):
            if yoda_output[k, 1].item() > 0.7:
                detected_cars.append(boxes[k])
                
        if detected_cars:
            detected_indices.append(idx)
            all_detected_cars.append(detected_cars)

        # Display ROIs predicted as cars
        image_detected_cars = image.copy()
        for box in detected_cars:
            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(image_detected_cars, pt1, pt2, color=(0, 255, 0), thickness=1)

        # cv2.imshow('Detected Cars', image_detected_cars)
        # cv2.waitKey(0)
        visualize_detected_and_ground_truth(image, detected_cars, ground_truth_cars)

        i += 1
        
    mean_iou = calculate_mean_iou(all_detected_cars, ground_truth_cars_list, detected_indices)
    print(f'Mean IoU over the test partition: {mean_iou}')

if __name__ == "__main__":
    main()