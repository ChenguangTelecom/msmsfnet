def crop_image(tensor, img_h, img_w):
    _,_,h,w=tensor.shape
    assert(img_h<=h and img_w<=w)
    h_s=(h-img_h)//2
    w_s=(w-img_w)//2
    tensor=tensor[:,:,h_s:h_s+img_h, w_s:w_s+img_w]
    return tensor
