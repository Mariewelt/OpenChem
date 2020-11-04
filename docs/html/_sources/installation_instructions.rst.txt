:github_url: https://github.com/Mariewelt/OpenChem

Installation instructions
=========================

General installation
--------------------

In order to get started you need to clone the repository to your local folder and install the requirements.
We recommend installation using Anaconda_::

    git clone https://github.com/Mariewelt/OpenChem.git
    cd OpenChem
    conda create --name OpenChem python=3.7
    conda activate OpenChem
    conda install --yes --file requirements.txt
    conda install -c rdkit rdkit nox cairo
    conda install pytorch torchvision -c pytorch
    pip install -e

.. _Anaconda: https://www.anaconda.com/
Installation with Docker
-------------------------

Alternative way of installation is with Docker. We provide a ``Dockerfile``, so that you can run your models in a container that already has all the necessary packages installed. You will also need nvidia-docker in order to run models on GPU.
If you are new to Docker, you could checkout this tutorial_.

.. _tutorial: https://opensource.com/business/14/7/guide-docker


First you need to install docker and nvidia-docker. Follow docker_ and nvidia-docker_ instructions.

.. _docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce
.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker

Then you need to clone the Openchem repository to your desired local folder by::

    git clone https://github.com/Mariewelt/OpenChem.git
    cd OpenChem

Then run the following command::

    docker build . -f Dockerfile

Execution of this command may take some time. After it's finished you will see a message similar to this one::

    Successfully built a40247366e78

Now you can start you docker container with nvidia-docker by running::

    nvidia-docker run -i -t a40247366e78

Inside you docker container you will have everything you need to start building models with OpenChem.
