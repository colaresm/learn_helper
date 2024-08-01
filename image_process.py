from basic_libs import *

def preprocess_image(image_path, image_size=224):

    img = cv2.imread(image_path)

    img = cv2.resize(img, (image_size, image_size))

    img = img / 255.0

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_img)
    img = enhancer.enhance(2.0)

    img = np.array(img) / 255.0
    processed_img_rgb = (img * 255).astype(np.uint8)

    return processed_img_rgb

