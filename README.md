# Yoga Model Hub

Welcome to the Yoga Model Hub! This repository is dedicated to providing machine learning models tailored for yoga-related applications. As of now, we offer a yoga pose classification model, but we're continually expanding. Stay tuned for more models and features in the near future!

## Current Features

- **Yoga Pose Classification Model**: Classify up to 82 different types of yoga poses. Outputs results in both English and Chinese.

## Setup

1. Clone this repository to your local machine.
2. Navigate to the root directory of the repository.
3. To set up the environment, run the following command:

```
pip install ."[serve]"
```

## Obtaining Model Weights
To use the Yoga Pose Classification Model effectively, you will need to obtain the necessary model weights. There are two primary methods to do this:

1. **Using DVC (Data Version Control)**
We use DVC to manage and version the model weights. This allows you to easily sync and retrieve the specific versions of the model weights used in this project. To pull the weights from our DVC-stored remote repository on Google Drive, follow these steps:
```bash
# Pull the model weights using DVC
dvc pull
```
2. **Manual Download**
If you prefer not to use DVC, or if you encounter issues with the automated process, you can manually download the model weights from the provided links. Below are the links to the weights for the ConvNeXt and DINOv2 models, which you can download and then place in the appropriate directory:

- [ConvNeXt Model Weights](https://drive.google.com/file/d/1EYtGMtITMzxYgm_hh6kTu3yxxMlVMJ9B/view?usp=sharing)
- [DINOv2 Model Weights](https://drive.google.com/file/d/1hNk97euC9S-Ce2dg7Omw5t0h_S0hcP1C/view?usp=sharing)

## Docker Setup
To run the application in a Docker container, follow these steps:

#### Building the Docker Image
Build the Docker image using the following command:
```go
make build
```
This command will create a Docker image with all the necessary dependencies installed.


#### Running the Docker Container
To start the application in a Docker container, use:
```go
make run
```
This will start the FastAPI server inside a Docker container.

#### Testing the API
You can test the API using a predefined curl command:
```go
make test-api
```
This command sends a POST request to the FastAPI server with a sample image for yoga pose classification.

## Inference (Yoga Pose Classification)

To classify yoga poses, you can use the provided sample code:

```python
from yogahub.models import YogaClassifier
from PIL import Image
import numpy as np

# Initialize the model
model = YogaClassifier()

# Example: Predict the yoga pose in an image
output = model.predict("example/test.png", convert_to_chinese=True)

# Sample output:
#{
#    "Target":"戰士二式(Virabhadrasana Ii)",
#    "PotentialCandidate":[
#        "舞王式(Natarajasana)",
#        "單腳向上延展式(Urdhva Prasarita Eka Padasana)",
#        "單腿站立伸展式(Utthita Padangusthasana)",
#        "戰士三式(Virabhadrasana Iii)"
#        ],
#    "Gesture":"站立"
# }

```

## License Information

The majority of the source code in this repository is licensed under the Apache License, Version 2.0. You can use, modify, distribute, and sell it both as part of a larger project and individually. The full license text can be found in the [LICENSE](LICENSE) file in the root of the repository.

### Special License Restrictions for the Yoga Pose Classification Model

Please note that the Yoga Pose Classification Model (`YogaClassifier`) has been trained using a dataset that is only licensed for non-commercial use, please refer to the [source](https://sites.google.com/view/yoga-82/home). As such, this model inherits these non-commercial restrictions.


## Future Work

I am always looking to expand the Yoga Model Hub. Expect more models and features related to yoga in the upcoming releases. Contributions and suggestions are welcome!
