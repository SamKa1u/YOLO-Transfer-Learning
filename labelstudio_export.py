from label_studio_sdk import Client

# Set your Label Studio details
LABEL_STUDIO_URL = "http://localhost:8080"  # Change this to your Label Studio instance
API_KEY = "<api_key>"  # Replace with your label-studio API key
PROJECT_ID = 2  # Replace with your actual project ID
EXPORT_TYPE = 'YOLO_WITH_IMAGES'

# Connect to the Label Studio API
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

# Get the project
project = ls.get_project(PROJECT_ID)

project.export_tasks(
    export_type=EXPORT_TYPE,
    download_resources=True,
    export_location='data.zip'
)