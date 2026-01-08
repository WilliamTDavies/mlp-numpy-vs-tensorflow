import numpy as np
import pygame
from PIL import Image
from scipy.ndimage import center_of_mass
import tensorflow as tf
from pathlib import Path
import sys

#Get path to MLP models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import src.numpy_mlp as npMLP
import src.tensorflow_mlp as tfMLP

def getImageData(screen):
    #Convert Pygame surface to 2D grayscale array
    a = np.array(screen.get_buffer()).reshape(screen.get_height(), screen.get_width(), 4)
    img = a[:, :, 0]

    #Threshold to get binary mask
    img = (img > 50).astype(np.uint8) * 255

    #Find bounding box of the digit
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((1, 784), dtype=np.float32)  #Blank if nothing drawn
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    #Crop and add small padding
    cropped = img[rmin:rmax+1, cmin:cmax+1]
    cropped = np.pad(cropped, ((4, 4), (4, 4)), mode='constant')

    #Resize to 20x20 while keeping aspect ratio
    pil_cropped = Image.fromarray(cropped)
    pil_cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
    cropped = np.array(pil_cropped)

    #Place into 28x28 image
    new_img = np.zeros((28, 28), dtype=np.uint8)
    h, w = cropped.shape
    top = (28 - h) // 2
    left = (28 - w) // 2
    new_img[top:top+h, left:left+w] = cropped

    #Center the digit using center of mass
    cy, cx = center_of_mass(new_img)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    new_img = np.roll(new_img, shiftx, axis=1)
    new_img = np.roll(new_img, shifty, axis=0)

    #Normalize and flatten
    new_img = new_img.astype(np.float32) / 255.0
    new_img = new_img.reshape(1, 784)
    return new_img

def preview(screen):
    #Create a preview of the 28x28 version of the image
    a = pygame.surfarray.array3d(screen)
    gray = a[:, :, 0]
    img = Image.fromarray(gray).resize((28, 28), Image.Resampling.LANCZOS)
    img_arr = np.array(img, dtype=np.uint8)
    scaled = np.kron(img_arr, np.ones((10, 10))).astype(np.uint8)
    surf = pygame.surfarray.make_surface(np.stack((scaled,)*3, axis=-1))
    return surf

def digitDraw():
    #Window initialisation
    pygame.init()
    screen = pygame.display.set_mode((560, 560))
    pygame.display.set_caption("Functionality Visualiser")

    #Separate canvas to preserve user drawing
    canvas = pygame.Surface((560, 560))
    canvas.fill((0, 0, 0))

    #Variable initialisation
    draw = False
    erase = False
    previewActive = False
    previewSurf = None
    running = True

    while running:
        for event in pygame.event.get():

            #Exit handling
            if event.type == pygame.QUIT:
                running = False

            #Detect mouse click
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    draw = True
                elif event.button == 3:
                    erase = True

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    draw = False
                elif event.button == 3:
                    erase = False

            #Drawing/erasing pixels
            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                if draw:
                    pygame.draw.circle(canvas, (255, 255, 255), (x, y), 20)
                elif erase:
                    pygame.draw.circle(canvas, (0, 0, 0), (x, y), 20)

            #Detect keyboard inputs
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    canvas.fill((0, 0, 0))
                elif event.key == pygame.K_RETURN:
                    return getImageData(canvas)
                elif event.key == pygame.K_p or event.key == pygame.K_ESCAPE:
                    previewActive = not previewActive
                    if previewActive:
                        previewSurf = preview(canvas)
                    else:
                        previewSurf = None

        #Draw canvas to main screen
        screen.blit(canvas, (0, 0))

        #Draw preview on top if active
        if previewActive and previewSurf:
            screen.blit(previewSurf, (280, 280))

        pygame.display.update()

    pygame.quit()

def setUp():
    #Ensure only the CPU is used to ensure greater similarities
    tf.config.set_visible_devices([], 'GPU')

    #Set random seeds for reproducible results
    tf.random.set_seed(0)
    np.random.seed(0)

def tfMLPDraw():
    #Create the full MLP
    X_train, y_train = tfMLP.getData()
    model, _ = tfMLP.modelCreation(X_train, y_train, 10, 32, 0.1) #Example hyperparameter values

    while True:
        #Generate the digit written by the user
        digit = digitDraw()

        #Predict the written digit
        p = model.predict(digit)
        prediction = np.argmax(p, axis=1)
        print(prediction)

def npMLPDraw():
    #Create the full MLP
    X_train, y_train = npMLP.getData()
    model, _ = npMLP.modelCreation(X_train, y_train, 10, 32, 0.1) #Example hyperparameter values

    while True:
        #Generate the digit written by the user
        digit = digitDraw()

        #Predict the written digit
        pred = npMLP.prediction(digit, model)
        print(pred)

def main():
    setUp()
    choice = input("\nDo you want to use the TensorFlow or NumPy MLP? (1/2)\n> ")
    while choice != '1' and choice != '2':
        print("\nInvalid choice, please enter either 1 (TensorFlow) or 2 (NumPy) only.")
        choice = input("\nDo you want to use the TensorFlow or NumPy MLP? (1/2)\n> ")
    if choice == '1':
        tfMLPDraw()
    else:
        npMLPDraw()

#To ensure the program is not accidentally run
if __name__ == "__main__":
    main()