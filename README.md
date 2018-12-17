# OpticalPy
OpticalPy is a dataset generator which use:
* background subtraction
* yolo object detection
* farneback's optical flow
For detect only movement object in a scene and extract only optical flow matrix (as numpy) and detect class with yolo.

## How to use it
Launch ```download_models.sh``` to create dataset folder and dowload yolo model
type ```python main.py``` to start the program

### Option
You can parse some argument for change the behaviour of application.
* ```-v``` select an input video
* ```-s``` select a path to save the output
* ```-show True``` show the result in realt time
* ```-y``` change yolo path folder
## Dependecy
* opencv-python
