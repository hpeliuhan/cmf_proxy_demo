FROM ubuntu:jammy
LABEL description="Small base image containing Python 3 and pywaggle."
LABEL maintainer="HPE <HAN.LIU@HPE.COM>"
LABEL url="https://github.com/waggle-sensor/plugin-base-images/tree/master/base"

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime

# install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    python3-pip \
    python3-opencv \
    curl \
    && rm -rf /var/lib/apt/lists/*

    WORKDIR /src

    # install waggle packages
    RUN pip3 install pywaggle[ALL]
    
   
    # Download the first matching video and save it as a fixed name
    # Set build argument for the video name
    ARG VIDEO_NAME="fire_sample1.mp4"
    
    # Download the specified video and save it as a fixed name
    RUN curl -L -o /src/video.mp4 https://raw.githubusercontent.com/hpeliuhan/cmf_test_example/main/$VIDEO_NAME
    
    # Download the model file
    ENV MODEL_TFLITE=/src/model.tflite
    RUN curl -L -o $MODEL_TFLITE https://raw.githubusercontent.com/hpeliuhan/cmf_test_example/main/model.tflite
    #update numpy
    RUN pip3 install ffmpeg numpy==1.23.5 opencv-python-headless tflite-runtime --upgrade
    
    # Copy source code into the container
    COPY src /src
    
    # Set the entrypoint
    ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]
    #ENTRYPOINT [ "python3", "inference.py" ]