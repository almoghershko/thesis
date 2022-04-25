FROM tensorflow/tensorflow:2.7.0-gpu

# copy the AWS credentials
COPY credentials /root/.aws/

# apt-get installs
RUN apt-get update \
    && apt-get install -y awscli \
    && apt-get install -y git 

# pip installs
RUN pip install --upgrade pip \
    && pip install pandas \
		&& pip install boto3 \
		&& pip install progressbar2 \
		&& pip install sfdmap \
		&& pip install GPUtil \
		&& pip install matplotlib \
		&& pip install matplotlib \
		&& pip install sklearn \
		&& pip install scipy \
		&& pip install astropy \
		&& pip install --upgrade awscli