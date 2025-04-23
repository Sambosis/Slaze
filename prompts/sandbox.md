Running this would remove essential packages like pandas, potentially breaking your entire codebase.



Isolating the execution environment

To prevent such disasters, we can create a sandbox — a separate, isolated environment — where code can be safely executed without risking the integrity of the main system.



The idea is to have two systems:



The Main System: Where we interact with the user or agent and manage all other tasks.

The Sandbox System: A separate, isolated environment (like a VM, cloud instance, or a different computer) where the generated code is executed. This system is designed to run code independently and return the results to the main system without affecting it.

Communication

To enable communication between these two systems, we'll use an API call. The main system will send the generated code and necessary files to the sandbox system via a web address or port. The sandbox will run the code, and once the execution is complete, it will send the results back to the main system. This is similar to how OpenAI's API operates.



Building a sandbox app

Now that we understand the importance of a secure sandbox, let's dive into creating one. By the end of this guide, we should have a sandbox app running inside a Docker container, allowing us to safely execute code without risking the integrity of our main system. This guide is not just about building the sandbox app — though you can certainly pull the final image from Docker Hub if you wish — but also about understanding how FastAPI, Docker images, and containers work together. This knowledge is essential for data scientists and engineers, as modern engineering systems increasingly rely on Docker containers and APIs.



Step 1: Set up the folder structure

First, let's create the following folder structure. We'll start by creating all the files and leaving them empty for now; we'll fill them in step by step



WorkFolder

├── session.py

├── test_session_and_main.py

└── sandbox

    ├── requirements.txt

    ├── main.py

    └── Dockerfile

Step 2: Install dependencies

Copy the following content into requirements.txt and then run the command below in your terminal after navigating to the sandbox folder:



# filename - requirements.txt

fastapi==0.112.0

pandas==2.2.2

uvicorn==0.30.6

numpy==2.0.1

python-multipart==0.0.9

requests==2.32.3

plotly==5.23.0

To install the dependencies, run:



pip install -r requirements.txt

Step 3: Create the FastAPI application

Next, copy the following code into the main.py file:



# filename - main.py

from fastapi import FastAPI, HTTPException

from fastapi import UploadFile, File

from fastapi.responses import JSONResponse

import pandas as pd

from io import BytesIO

import json

import pickle



# Instantiate FastAPI application

app = FastAPI()



# Initialization of dictionaries to upload files and save results

df_dict = {}

result_dict = {}





@app.post("/uploadfile/{file_name}")

async def upload_file(file_name: str, file: UploadFile = File(...)):

    """

    This function manages the upload of files. 

    It saves the content of the file into a pandas dataframe 

    into the df_dict dictionary

    """

    # Read file content

    content =  await file.read()

    df = pd.read_csv(BytesIO(content))

    df_dict[file_name] = df

    

    return {"message": f"File {file_name} has been uploaded"}



@app.get("/files/")

async def get_files():

    """

    This function returns a list with the names of all uploaded files.

    """

    return {"files": list(df_dict.keys()) }



@app.get("/downloadfile/{file_name}")

async def download_file(file_name: str):

    """

    This function manages the download of files. 

    It returns the content of a specific file converted into a Json format.

    """

    df = df_dict.get(file_name, pd.DataFrame())

    return JSONResponse(content=df.to_json(orient="records"))



@app.post("/execute_code/")

async def execute_code(request_body: dict):

    """

    This function manages the execution of Python code 

provided by the user in the request body.

    In the end, it returns the results in a Json format.

    """

    # Initialize local dictionary

    local_dict = {}

    code = request_body.get("code")

    try:

        exec(code, df_dict, local_dict)

        result_dict.update(local_dict)



        # Convert any non-json serializable items into string using str().

        # This is required to send the data back using api and we 

        # can not send dataframes 

        # directly back

        for k, v in local_dict.items():

            if isinstance(v, pd.DataFrame):

                local_dict[k] = v.to_json()

        local_dict = {k: str(v) if not isinstance(v, \

        (int, float, bool, str, list, dict, pd.DataFrame)) \

         else v for k, v in local_dict.items()}\

            

    except Exception as e:

        raise HTTPException(status_code=400, detail=str(e))



    # Serialize local_dict to a JSON-formatted string

    local_dict = json.dumps(local_dict)

    return local_dict



@app.get("/clear_state/")

async def clear_state():

    """

    This function manages the reset of the dictionaries so 

    that it does not intefares with previous uploaded files.

    """

    df_dict.clear()

    result_dict.clear()

    return {"message": "State has been cleared"}

You might be wondering why we're using FastAPI to execute code when we could just run it locally with the exec method. The answer is that we don't want to run potentially unsafe code on our own machines. Instead, we'll execute it on another system—like a separate machine or a container—so that any issues won't affect our main environment. FastAPI allows us to send code and data to this separate system and receive the results safely via an API.



To run this application, use the following command in your terminal:



uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Step 4: Create a python wrapper

While you can use requests.post or requests.get directly to interact with the API, it's more elegant to create a Python wrapper that abstracts these calls into methods. Copy the following code into the session.py file:



# filename- session.py

import requests

import pandas as pd

from typing import Any

from io import StringIO



class CodeSession:

    def __init__(self, url: str) -> None:

        self.url = url

    

    def upload_file(self, file_name: str, data: pd.DataFrame) -> Any:

        data_str = data.to_csv(index=False)

        files = {"file": StringIO(data_str)}

        return requests.post(f'{self.url}/uploadfile/{file_name}', files=files).json()

    

    def get_files(self) -> Any:

        return requests.get(f'{self.url}/files/').json()

    

    def download_file(self, file_name: str) -> pd.DataFrame:

        response = requests.get(f'{self.url}/downloadfile/{file_name}').json()

        return pd.read_json(response)



    def execute_code(self, code: str) -> Any:

        return requests.post(f'{self.url}/execute_code/', json={"code": code}).json()



    def clear_state(self) -> Any:

        return requests.get(f'{self.url}/clear_state/').json()

This wrapper makes it easier to interact with the sandbox app in a Pythonic way, allowing you to focus on writing and testing your code without getting bogged down in API calls.



Step 5: Testing the sandbox app

To ensure that our FastAPI sandbox app is working correctly, we need to test it. Here's how you can do that:



Copy the following code into a file named test_session_and_main.py



# filename test_session_and_main.py



import sys

from session import CodeSession

import pandas as pd



# Get the port number from command-line arguments

if len(sys.argv) != 2:

    print("Usage: python test_session_and_main.py <port>")

    sys.exit(1)



port = sys.argv[1] # getting port number from system arguments 

base_url = f'http://127.0.0.1:{port}'

session = CodeSession(base_url)



# Prepare some sample data

data = pd.DataFrame({

    'A': [1, 2, 3],

    'B': [4, 5, 6]

})



# Use the session to upload a file

upload_resp = session.upload_file('df', data)

print("Upload Response:", upload_resp)



# Use the session to get the list of files

files = session.get_files()

print("Files in Sandbox:", files)



# Use the session to download a file

downloaded_df = session.download_file('df')

print("Downloaded DataFrame:")

print(downloaded_df)



# Use the session to execute some code on the server

result = session.execute_code("result = df['A'].sum()")

print("Execution Result:", result)

Execute the script by running(we need to go to the parent directory by using cd ..) :





python test_session_and_main.py 8000

This will interact with the FastAPI app running locally, and you should see output confirming that each operation — uploading, listing, downloading, and executing code — is working correctly.



By running this test, you'll confirm that your FastAPI app is functioning as intended and is ready for further use or deployment.



Detour- what is a container?

A container is a lightweight, standalone, and executable software package that includes everything needed to run a piece of software: the code, runtime, system tools, libraries, and settings. Containers isolate software from its environment and ensure that it works uniformly despite differences in operating systems and underlying infrastructure.



Why is virtualization required?

Virtualization allows multiple operating systems or applications to run on a single physical machine by creating virtual versions of hardware, software, or storage. This is essential because it provides a way to isolate applications from one another, reducing the risk of conflicts and enhancing security.



Why is virtualization important for our case?

In our case, virtualization is crucial because we need to run potentially unsafe code. Without virtualization, running this code directly on your machine could lead to system crashes, data loss, or other severe issues. Virtualization, through containers, allows us to isolate the execution environment, ensuring that any harmful effects are contained within the virtual environment and do not affect your main system.



What is an image?

An image is a blueprint or a snapshot of a container. It contains everything needed to run an application, including the operating system, application code, libraries, and dependencies. Images are immutable, meaning once they are created, they don't change. They can be stored and reused, ensuring consistency across different environments.



How is an image made and structured in layers?

A Docker image is made by writing a Dockerfile, a script that contains a series of instructions for assembling an image. These instructions might include copying files, setting environment variables, and installing software packages. Each instruction in the Dockerfile creates a new layer in the image. Layers are stacked on top of each other, and Docker caches these layers to speed up the build process. If any part of the image changes, only the affected layers are rebuilt, which saves time and resources.



How does a container run an image?

When you run a container, it creates an instance of an image. This means the container is a running version of the image, with its own file system, network interface, and process space, but it still shares the kernel of the host operating system. Containers are lightweight because they don't need a full-fledged operating system; they only need the resources specified in the image layers.



Using images in kubernetes pods

Kubernetes is a powerful orchestration tool that manages containerized applications. It allows you to deploy, scale, and manage containers across a cluster of machines. A Pod in Kubernetes is the smallest and simplest unit that you can create or deploy. It can contain one or more containers, all sharing the same network and storage resources.



When you deploy an image in a Kubernetes Pod, Kubernetes schedules it on one of the cluster's nodes, ensuring it has the necessary resources to run. If the container fails, Kubernetes can automatically restart it or even replace it with a new container based on the same image, providing robust scalability and reliability in production environments.



Why move our FastAPI app to a container?

We've created a FastAPI application with endpoints for uploading and downloading files, executing code, and clearing the state. However, we're currently running this app on our local machine, which means we're still vulnerable to the risks we discussed earlier.



To truly implement a secure sandbox, we need to package this FastAPI app into a container. A Docker container acts like a separate machine within your machine. Running potentially unsafe code in this container won't harm your main system. Even if something goes wrong and the container gets corrupted, it's easy to revive it by recreating it from the same image. Moreover, in production environments, there are automated systems to repair or replace damaged containers, ensuring minimal downtime and consistent performance.



Step 6: Writing the Dockerfile

A Dockerfile is essentially a recipe that tells Docker how to build an image for our application. Here's the Dockerfile we'll use for our sandbox app:



# filename - Dockerfile

FROM python:3.11-slim



WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

Breaking Down the Dockerfile:



FROM python:3.11-slim: This line sets the base image as Python 3.11. The slim version is a minimal image that includes just the essentials, making it lightweight and efficient.

WORKDIR /app: This creates and sets the working directory inside the container where all subsequent commands will be run.

COPY requirements.txt requirements.txt: This copies the requirements.txt file from your local machine to the container.

RUN pip3 install -r requirements.txt: This installs all the necessary Python packages listed in requirements.txt inside the container.

COPY . .: This copies the rest of your application's files into the container.

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]: This specifies the command to start the FastAPI app using Uvicorn, setting it to listen on port 8000 of the container.

With this Dockerfile, we can create an image that contains our entire FastAPI app, fully self-contained and ready to run.



Step 7: Building and running the image

Now, let's build the Docker image and run it in a container:



Note: In this setup, we are using nerdctl, which is a command-line interface for containerd. Containerd is a lightweight container runtime that handles the low-level details of running and managing containers. I got this along with Rancher Desktop. Alternatively, you can use Docker, which provides a user-friendly interface for container management. If you're using Docker Desktop instead of Rancher Desktop, you can replace nerdctl with docker in all the commands provided below.



Build the image(on navigating to sandbox folder):



nerdctl build -t code_session .

This command creates an image named code_session based on the Dockerfile



Run the container:



nerdctl run -d -p 8080:8080 --name sandbox code_session

This command starts a container named sandbox from the code_session image. It maps port 8000 inside the container to port 8080 on your machine, allowing you to access the app at http://localhost:8080.



Step 8: Monitoring the container

If you want to see the logs or monitor what's happening inside the container, you can use the following command



nerdctl logs -f sandbox

This will show you the real-time logs from the sandbox container.



Directly pulling the image

If you prefer not to build the image yourself, you can directly pull and run the pre-built image from Docker Hub:



Pull the image:



nerdctl pull shrishml/code_session

Run the container:



nerdctl run -d --name sandbox -p 8080:8000 shrishml/code_session

And that's it! You now have a fully functional sandbox environment running safely inside a Docker container.



To test the FastAPI sandbox app, you can use the following command:



python test_session_and_main.py 8080

In this case, the port 8080 is used, which is connected to the port 8080 of the Docker container running your FastAPI app. This is different from the port 8000 where your FastAPI app might have been running locally before. Since you have now set up the app inside a container on port 8080, you can close any previous instances of the FastAPI app running on port 8000.



This command will interact with the FastAPI sandbox app inside the container, allowing you to test file uploads, downloads, and code execution through the specified port.