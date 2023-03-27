
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import keras.utils as image
model = tf.keras.models.load_model('./models/global_model.h5')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Load the CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# # Normalize the input data
# x_test = x_test.astype('float32') / 255.

# # Choose a random image from the test set
# index = np.random.randint(0, len(x_test))
# image = x_test[index]

# # Display the image
# plt.imshow(image)
# plt.show()

# # Reshape the image to match the input shape of the model
# image = np.expand_dims(image, axis=0)

img = image.load_img('./static/images/OIP.jpg',target_size=(32,32,3))
x = image.img_to_array(img)/255
x = np.expand_dims(x,axis=0)

# Use the predict function to obtain the model's output
y_pred_task1, y_pred_task2 = model.predict(x)

# Display the model's output
print('Task 1 predictions:', y_pred_task1)
print('Task 2 prediction:', y_pred_task2)
# Define the class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Convert the task 1 prediction to text
task1_class_index = np.argmax(y_pred_task1)
task1_prediction_text = 'Task 1 prediction: {}'.format(class_names[task1_class_index])

# Convert the task 2 prediction to text
task2_prediction_text = 'Task 2 prediction: {}'.format('Yes' if y_pred_task2 > 0.5 else 'No')

# Display the prediction texts
print(task1_prediction_text)
print(task2_prediction_text)