# VAR-Robotics-Project-2026

## Image to Drone movement

### Repository Layout

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