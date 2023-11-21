import torch
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

def transform_function(roi):
    roi = cv2.resize(roi, (150, 150))
    roi_tensor = transforms.ToTensor()(roi)
    roi_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(roi_tensor)
    return roi_tensor

def calculate_mean_iou(detected_cars, ground_truth_cars_list):
    total_iou = 0
    total_detected_cars = len(detected_cars)

    for detected_image_cars, ground_truth_cars in zip(detected_cars, ground_truth_cars_list):
        max_iou = 0

        for detected_car in detected_image_cars:
            for ground_truth_car in ground_truth_cars:
                # Ensure that the input arguments are properly formatted as (box, box)
                iou = Anchors().calc_IoU(detected_car, ground_truth_car)
                max_iou = max(max_iou, iou)

        total_iou += max_iou

    mean_iou = total_iou / total_detected_cars if total_detected_cars > 0 else 0
    return mean_iou


def main(args):
    print('running YODA test...')

    input_dir = None
    if args.i is not None:
        input_dir = args.i

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(input_dir, training=False)
    num_classes = 2
    model = YodaClassifier(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.load_state_dict(torch.load('yoda_classifier_with_early_stopping_LR00001B96.pth'))

    ground_truth_cars_list = []  
    all_detected_cars = []  
    detected_indices = []
    for item in enumerate(dataset):

        idx = item[0]
        image = item[1][0]
        label = item[1][1]

        idx = dataset.class_label['Car']
        ground_truth_cars = dataset.strip_ROIs(class_ID=idx, label_list=label)
        ground_truth_cars_list.append(ground_truth_cars)

        anchor_centers = Anchors().calc_anchor_centers(image.shape, Anchors().grid)
        ROIs, boxes = Anchors().get_anchor_ROIs(image, anchor_centers, Anchors().shapes)

        roi_batch = torch.stack([transform_function(roi) for roi in ROIs])
        model.eval()
        with torch.no_grad():
            yoda_output = model(roi_batch.to(device))

        detected_cars = []
        for k in range(len(boxes)):
            if yoda_output[k, 1].item() > 0.9:
                detected_cars.append(boxes[k])
                
        if detected_cars:
            detected_indices.append(idx)
            all_detected_cars.append(detected_cars)

        # visualize_detected_and_ground_truth(image, detected_cars, ground_truth_cars)
 
    mean_iou = calculate_mean_iou(all_detected_cars, ground_truth_cars_list)
    print(f'Mean IoU over the test partition: {mean_iou}')

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')

    args = argParser.parse_args()
    main(args)