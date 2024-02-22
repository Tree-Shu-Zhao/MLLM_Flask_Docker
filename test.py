import requests

# URL of your Flask route
url = 'http://localhost:33670/vicuna'

# Path to the image file
image_path = '/scratch/shuzhao/Dataset/USGG/GQA/images/10.jpg'

# Text to send
text = 'Introduce PennState.'

# Open the image file in binary mode
with open(image_path, 'rb') as image_file:
    # Create a dictionary with the text and image file
    files = {'image': image_file}

    # Send a POST request to the Flask route with the text and image
    response = requests.post(url, data={"text": text}, files=files)

# Print the response from the server
print(response.text)

