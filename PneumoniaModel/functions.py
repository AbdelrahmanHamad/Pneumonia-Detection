def read_images(folder_path , label , size = (128, 128)):
    import os
    import cv2
    import numpy as np

    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
            labels.append(label)

    return np.array(images) , np.array(labels)

def mergain(Images1 , Images2 , labels1 , labels2):
    import numpy as np
    return np.concatenate((Images1 , Images2) , axis = 0) , np.concatenate((labels1 , labels2) , axis = 0)
    
def split_data(Images , labels , test_size = 0.2):
    import numpy as np
    from sklearn.model_selection import train_test_split
    return train_test_split(Images , labels , test_size = test_size , shuffle = True , stratify = labels)

def create_data_augmentation():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    data_gen = ImageDataGenerator(

        rotation_range=20,       # Random rotations from 0 to 40 degrees
        # width_shift_range=0.2,   # Random horizontal shifts up to 20% of the total width
        # height_shift_range=0.2,  # Random vertical shifts up to 20% of the total height
        # shear_range=0.2,         # Shear transformations
        zoom_range=0.2,          # Random zooming up to 20%
        horizontal_flip=True,    # Random horizontal flips
        fill_mode='nearest'      # Strategy to fill newly created pixels after a rotation or width/height shift
    )
    return data_gen



def show_rotated_image(image_path):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    # Load the image
    image = Image.open(image_path)
    
    # Generate a random angle between 0 and 40 degrees
    angle = np.random.uniform(0, 40)
    
    # Rotate the image
    rotated_image = image.rotate(angle)
    
    # Display the original and rotated images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(rotated_image)
    ax[1].set_title(f"Rotated Image by {angle:.2f} Degrees")
    ax[1].axis('off')
    
    plt.show()

def show_zoomed_image(image_path):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the image
    image = Image.open(image_path)
    
    # Generate a random zoom factor between 0.6 and 1.4
    zoom_factor = np.random.uniform(0.6, 1.4)
    
    # Calculate the new dimensions
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    # Resize the image using the zoom factor
    zoomed_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Center the crop to the original size
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height
    cropped_zoomed_image = zoomed_image.crop((left, top, right, bottom))
    
    # Display the original and zoomed images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(cropped_zoomed_image)
    ax[1].set_title(f"Zoomed Image by a factor of {zoom_factor:.2f}")
    ax[1].axis('off')
    
    plt.show()

def show_flipped_image(image_path):
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the image
    image = Image.open(image_path)
    
    # Perform horizontal flip
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Display the original and flipped images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(flipped_image)
    ax[1].set_title("Horizontally Flipped Image")
    ax[1].axis('off')
    
    plt.show()

# Example usage: show_flipped_image("path_to_your_image.jpg")
# This function expects a path to an image file as input and will display the original and a horizontally flipped image side by side.
