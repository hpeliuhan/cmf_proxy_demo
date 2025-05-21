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

    
    # Copy source code into the container
    COPY src /src
    
    # Set the entrypoint
    #ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]
    ENTRYPOINT [ "python3", "main.py" ]