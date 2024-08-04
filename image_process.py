from basic_libs import *
 

def enhance_retinal_imag(image, alpha=4, beta=-4, offset=128):
    
    kernel_size = (9,9) 

    smoothed_image = cv2.GaussianBlur(image, kernel_size, sigmaX=10, sigmaY=0)

    image_float = image.astype(np.float32)

    smoothed_float = smoothed_image.astype(np.float32)

    enhanced_image = alpha * image_float + beta * smoothed_float + offset


    enhanced_image = np.clip(enhanced_image, 0, 255)


    enhanced_image = enhanced_image.astype(np.uint8)

    return  enhanced_image


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, k=1.0):

    blurred_image = image
    
 
    image_float = image.astype(np.float32)
    blurred_float = blurred_image.astype(np.float32)
    
 
    mask = image_float - blurred_float
    
 
    mask = mask * k
    
  
    sharp_image = image_float + mask
    
 
    sharp_image = np.clip(sharp_image, 0, 255)
    
 
    sharp_image = sharp_image.astype(np.uint8)


    return sharp_image

def apply_gamma(image, gamma_value=0.5047):
 
    def gamma_transformation(image, gamma):
    
        gamma_corrected = np.power(image, gamma)
        return  gamma_corrected

    def contrast_stretching(image):
      
        Imin = np.min(image)
        Imax = np.max(image)
        
       
        stretched_image = 255 * (image - Imin) / (Imax - Imin)
        return np.uint8(stretched_image)
    
     
    gamma_image = gamma_transformation(image, gamma_value)
    
    
    final_image = contrast_stretching(gamma_image)
    
    return final_image


def preprocess_image(image_path):
  image= cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

  #image= cv2.resize(image, (224, 224)) 

  #image=enhance_retinal_imag(image)

 # image = unsharp_mask(image)
  
  #image=apply_gamma(image)
 

  return image/255.0