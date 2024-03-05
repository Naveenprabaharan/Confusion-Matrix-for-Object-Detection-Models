# all image working- V1
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from IPython.display import Image as imgImage
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

ground_truth_annotations_l = []
predicted_labels_l = []
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('.//object'):
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append((ymin, xmin, ymax, xmax))

    return boxes

def get_ground_truth_annotations(xml_path):
    """Read ground truth annotations from XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        annotations.append({'name': name})

    return annotations

images = os.listdir('/content/test_')
print(len(images))
isExist = os.path.exists('/content/raw_test_image_output')
if not isExist:
  os.mkdir('/content/raw_test_image_output')

confusion_matrix_data = {'true_labels': [], 'predicted_labels': []}

for img in images:
    image_path = '/content/test_/' + img
    xml_path = '/content/test_/' + img.replace('.jpg', '.xml')  # Replace with your ground truth annotations path

    try:
      if image_path[-4:]=='.jpg':
        print('image_path : ',image_path)
        raw_output_image_path = f'/content/raw_test_image_output/{img}'
        pil_img = imgImage(filename=raw_output_image_path)
        display(pil_img)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        detection_scores = detections['detection_scores'][0].numpy()
        # detection_classes = detections['detection_classes'][0].numpy() + 1
        detection_classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
        
        pred_bbox = []
        for i in range(len(detections['detection_boxes'][0].numpy())):
          detection_bbox = detections['detection_boxes'][0][i].numpy()*512
          pred_bbox.append(detection_bbox.astype(int))
        pred_bbox = np.array(pred_bbox)
        # print('pred_bbox : ',pred_bbox)
        # print('detection_scores : ',detection_scores)
        # print('detection_classes : ',detection_classes)


        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        # Get ground truth annotations
        ground_truth_annotations = get_ground_truth_annotations(xml_path)
        # print('ground_truth_annotations: ',ground_truth_annotations)
        ground_truth_bboxes = parse_xml(xml_path)
        # print('ground_truth_bboxes: ',ground_truth_bboxes)
        # Flatten the ground truth and predicted labels for the confusion matrix
        true_labels = [annotation['name'] for annotation in ground_truth_annotations]
        predicted_labels = [category_index[int(class_id)]['name'] for class_id in detection_classes]
        # print('predicted_labels : ',predicted_labels)
        # Update the confusion matrix data
        confusion_matrix_data['true_labels'].extend(true_labels)
        # print(true_labels)
        confusion_matrix_data['predicted_labels'].extend(predicted_labels)
        # print(predicted_labels)

        # finding best bounding box
        # Initialize lists to store the best bounding boxes and their IoU scores

        best_boxes = []

        best_ious = []
        best_index = []
        # Find the best bounding box for each true box

        for true_box in ground_truth_bboxes:

          max_iou = 0

          best_box = None

          for candidate_box in pred_bbox:

              iou = calculate_iou(true_box, candidate_box)

              if iou > max_iou:

                  max_iou = iou

                  best_box = candidate_box
                  index = np.where((pred_bbox == best_box).all(axis=1))[0][0]

          best_index.append(index)
          best_boxes.append(best_box)
          best_ious.append(max_iou)

        # print('oo',best_boxes,best_ious,best_index)
        pred_bbox = np.array(pred_bbox)[best_index]
        predicted_labels = np.array(predicted_labels)[best_index]
        detection_scores = np.array(detection_scores)[best_index]
        # print('detection_scores : ',predicted_labels)
        precision, recall = calculate_precision_recall(np.array(ground_truth_bboxes), np.array(ground_truth_annotations), np.array(pred_bbox), np.array(predicted_labels), np.array(detection_scores))

        print("Precision:", precision)
        print("Recall:", recall)

        ground_truth_annotations_i = []
        # adding label for confusion matrix
        for id in range(len(ground_truth_annotations)):
          ground_truth_annotations_i.append(ground_truth_annotations[id]['name'])

        
        # print('ground_truth_annotations_i : ',ground_truth_annotations_i)
        # print('predicted_labels : ',predicted_labels)
        l = ['Goggle','Helmet','Person','Shoe','Vest']
        ground_truth_annotations_i = sorted(ground_truth_annotations_i)
        predicted_labels = sorted(predicted_labels)
        print('ground_truth_annotations_i : ',ground_truth_annotations_i)
        print('predicted_labels : ',predicted_labels)
        conf_matrix = confusion_matrix(ground_truth_annotations_i, predicted_labels,labels=l)
        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=l)
        # disp.plot()
        # plt.show()
        print("Confusion Matrix:")
        print(conf_matrix)



        # adding label for confusion matrix
        for id in range(len(ground_truth_annotations)):
          ground_truth_annotations_l.append(ground_truth_annotations[id]['name'])

        predicted_labels_l.extend(predicted_labels)

        label_id_offset = 1
        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.5,
              agnostic_mode=False,
        )

        plt.figure(figsize=(12,16))
        plt.imshow(image_np_with_detections)
        plt.show()
        save_path = f'/content/raw_test_image_output/{img[:14]}'
        print(save_path)
        plt.savefig(f'/content/raw_test_image_output/{img}')




    except Exception as inst:
        print(inst)


print('predicted_labels : ',predicted_labels_l)
print('ground_truth_annotations_l : ',ground_truth_annotations_l)
conf_matrix = confusion_matrix(ground_truth_annotations_l, predicted_labels_l)
print("Confusion Matrix:")
print(conf_matrix)