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

def model_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def class_frequency(labels , label_name):
    import numpy as np
    import matplotlib.pyplot as plt
    unique , counts = np.unique(labels , return_counts = True)

    plt.bar(label_name , counts)
    plt.show()


def classifiction_report(y_test , y_pred):
    from sklearn.metrics import classification_report
    
    y_pred_results = []
    for i in y_pred:
        if i > 0.5:
            y_pred_results.append(1)
        else:
            y_pred_results.append(0)

    print(classification_report(y_test , y_pred_results))


def confusion_matrix(y_test , y_pred , label_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    y_pred_results = []

    for i in y_pred:
        if i > 0.5:
            y_pred_results.append(1)
        else:
            y_pred_results.append(0)

    conf = confusion_matrix(y_test , y_pred_results)
    sns.heatmap(conf, annot = True , fmt = 'g' , xticklabels = label_name , yticklabels = label_name)
    plt.show()