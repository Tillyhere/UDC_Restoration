# Pyramid Scheme for Efficient Under-Display Camera Image Restoration
Welcome to my project repository! This archive contains my previous work, primarily focused on techniques for under-display camera (UDC) image restoration.
The code is built upon [DAGF](https://github.com/varun19299/deep-atrous-guided-filter) and [LPTN](https://github.com/csjliang/LPTN), and was completed in 2022. 
Please note that it may be out of date due to the rapid advancements in UDC restoration. Nonetheless, I am grateful for every line of code being here.

## Motivation
Image attributes such as lightness and color are mostly linked to the low-frequency component. 
![image](https://github.com/Tillyhere/UDC_Restoration/assets/48005193/0213d1d5-81e6-4390-9ddc-f6a8acb82faf)

## Pyramid Scheme
Network Overview (Laplacian Pyramid)
![image](https://github.com/Tillyhere/UDC_Restoration/assets/48005193/85989902-ad21-4ff3-9017-b6e9973ecd86)

 Proposed module incorporates pyramid channels and pyramid feature receptive fields.
![image](https://github.com/Tillyhere/UDC_Restoration/assets/48005193/3943e5ac-e29b-4b9a-97c1-63591bcae135)


A rough report can be found here. [Baidu Disk](https://pan.baidu.com/s/1RvKKIXEXVfljv_3kupyBlw?pwd=3407) (verification code: 3407)


Training

For POLED
'''
 python train_lplr_VII.py with poled_lp_lr_VII
'''

