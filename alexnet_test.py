import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('alexnet_tomato_leaf_disease_model.h5')

# Specify the path to the folder containing test images
test_folder_path = os.path.join(os.path.dirname(__file__), 'test')

# Load and prepare the class indices from the training generator
train_datagen = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    os.path.join(os.path.dirname(__file__), 'training'),  # Update to your actual training dataset folder
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=False
)

# Map class indices to class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Iterate through all files in the folder
for filename in os.listdir(test_folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Construct the full path to the image
        test_image_path = os.path.join(test_folder_path, filename)

        # Load and preprocess the test image
        test_image = image.load_img(test_image_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0

        # Make predictions
        predictions = model.predict(test_image)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)

        # Map the predicted class index to the class label
        predicted_class_label = class_labels.get(predicted_class, "Unknown")

        # Print the predicted class for each image
        print(f"Image: {filename}, Predicted class: {predicted_class_label}")
