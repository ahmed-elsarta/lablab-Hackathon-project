# lablab-Hackathon-project

## Demo
https://github.com/ahmed-elsarta/lablab-Hackathon-project/assets/93448764/9c4a5cc0-f3a5-446c-ac6d-a1b005ae8891

# About ":white_check_mark:NurseGPT"
![2023-06-17 (30)](https://github.com/ahmed-elsarta/lablab-Hackathon-project/assets/93448764/16a71b98-b083-4b97-9ca0-b3141dcda2ce)
An interactive medical chat-bot based on Streamlit and Langchain Python Frameworks, the chat-bot responds in the same language as the user, it is supposed to assess the risk of cardiovascular diseases for male users and maternal diseases for female users, as low or high risk, the chat-bot asks the user for the symptoms or medical information and the data is then extracted from the input text (prompt) using Langchain's built in module "Object", after the data or required information is extracted from the input text the information is fed into a pre-made machine learning model depending on the gender of the user, maternal model for females and cardiovascular model for males, where the value of risk is computed from the model and upon its value the bot would respond with high or low risk to the user.

## Snapshots
- English Chatbot
![2023-06-17 (29)](https://github.com/ahmed-elsarta/lablab-Hackathon-project/assets/93448764/619e5903-bc2f-4056-a123-ae4e1b07ba24)

- Arabic Chatbot
![2023-06-17 (27)](https://github.com/ahmed-elsarta/lablab-Hackathon-project/assets/93448764/83058dee-bb78-4929-9ca4-ca1be1f6a455)

## Frameworks Used
- Python's Streamlit Framework (UI)
- Python's Langchain Framework (Chat-bot)

## Run The App
- Download Python 3
- Download VS code

 1-  Clone The project
  ```ruby
  git clone https://github.com/ahmed-elsarta/lablab-Hackathon-project.git
  ```
 2-  Install packages
  ```ruby
  pip install -r requirements.txt
  ```
 3-  Run the app
  ```ruby
  streamlit run streamlit_app.py
  ```
