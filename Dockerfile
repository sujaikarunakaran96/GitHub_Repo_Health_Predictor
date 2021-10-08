FROM python:3.6-slim
COPY ./dash_app_v4.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./Spring_2021_Model3.h5 /deploy/
COPY ./Test_Weekly_Activity_Score_R_local.csv /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "dash_app_v4.py"]






