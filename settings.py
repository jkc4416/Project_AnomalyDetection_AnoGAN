import shutil
import os

def settings():
    target_folder = 'C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/dataset/train/train'
    
    try:
        shutil.rmtree(target_folder)
        print("remove files!")
    except Exception as e:
        print(e)
        pass

    try:
        if not(os.path.isdir(target_folder)):
            os.makedirs(os.path.join(target_folder))
    except OSError as e:
        print(e)

    target_folder = 'C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/dataset/valid/valid'
    try:
        shutil.rmtree(target_folder)
        print("remove files!")
    except Exception as e:
        print(e)
        pass

    try:
        if not (os.path.isdir(target_folder)):
            os.makedirs(os.path.join(target_folder))
    except OSError as e:
        print(e)


    target_folder = 'C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/dataset/test/test'
    try:
        shutil.rmtree(target_folder)
        print("remove files!")
    except Exception as e:
        print(e)
        pass

    try:
        if not (os.path.isdir(target_folder)):
            os.makedirs(os.path.join(target_folder))
    except OSError as e:
        print(e)


    target_folder = 'C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/saved_model'
    try:
        shutil.rmtree(target_folder)
        print("remove files!")
    except Exception as e:
        print(e)
        pass

    try:
        if not(os.path.isdir(target_folder)):
            os.makedirs(os.path.join(target_folder))
    except OSError as e:
        print(e)


    target_folder = 'C:/Users/posco/PycharmProjects/LSD_AnoGAN_pytorch/result'
    try:
        shutil.rmtree(target_folder)
        print("remove files!")
    except Exception as e:
        print(e)
        pass

    try:
        if not(os.path.isdir(target_folder)):
            os.makedirs(os.path.join(target_folder))
    except OSError as e:
        print(e)