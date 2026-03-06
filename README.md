# VAR-Robotics-Project-2026

## Image to Drone movement
Takes image, finds the main object of the image, samples num_drones amount of points to represent that image the most, then from the acheived coordinates, move the drones so that it forms the object.

### Pipeline
```
Image -> Extract Features -> Priority Map -> Sample Points -> JSON Coordinates file -> 
```

### Repository Layout
```
VAR-ROBOTICS-PROJECT-2026
├── Dockerfile
├── README.md
├── brain
│   ├── main_brain.py
│   └── src
│       ├── feature_extractor.py
│       ├── features.py
│       └── point_sampler.py
├── brawn
│   ├── controllers
│   │   ├── drone_member
│   │   │   ├── drone_member.py
│   │   │   └── mavic_logic.py
│   │   └── drone_supervisor
│   │       └── supervisor.py
│   └── worlds
├── data
│   ├── coordinates
│   ├── input_images
│   └── outputs
└── requirements.txt
```

### Key Features

### Build & Run
