FROM nvcr.io/nvidian_general/tensorflow:18.05-py3

RUN apt-get update -y

# ubuntu dependencies for flex gym 
RUN apt-get install -y tk-dev python3-tk zlib1g-dev

# Python dependencies for flex gym 
RUN pip install autolab_core gym pandas

# Python dependencies for NvBaselines. We install here instead of in pip later b/c 1) use docker cache and 2) avoid uneeded gym envs (e.g. atari, mujoco, etc)
RUN pip install tqdm joblib zmq dill progressbar2 mpi4py cloudpickle click

# Copy baselines
COPY NvBaselines NvBaselines
RUN pip install --no-deps NvBaselines/

# Copy current build
COPY rbd rbd
RUN pip install rbd/demo/gym

# Copy start up script
COPY entry.sh /entry.sh
RUN chmod +x /entry.sh

# Run start script
CMD ["/entry.sh"]
