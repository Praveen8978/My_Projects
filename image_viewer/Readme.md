# **Image Viewer (PPM – Binary P6 Format)**

## **Description**
This project is a lightweight image viewer capable of displaying **PPM (Portable Pixmap) images**, specifically the **binary P6** variant.  
You can determine the format by inspecting the first line of a PPM file:

- `P6` → Binary (supported)  
- `P3` → ASCII (not currently supported)

The viewer is implemented in C++ and uses **SDL2** for rendering.

---

## **Prerequisites**
This project depends on:

### **Libraries**
- **SDL2** — for window creation and rendering
- **C++ Standard Library**
  - `fstream`
  - `string`
  - `sstream`

### **Installing SDL2 on Ubuntu**
```bash
sudo apt install libsdl2-dev libsdl2-2.0-0 -y
```
## **Acknowledgements**
This project was inspired by the tutorial:  
**[Coding an Image Viewer in Pure C](https://youtu.be/sItRLFjbqvo?si=Z-NaORO9OOqaTgT_)**


## Future Improvements
>Add additional File format Support  
>Add GUI support to upload Images
