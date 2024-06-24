# Safety Helmet Detection
View our app here ➡️ 👷 [Safety Helmet Detection App](https://helmet-det.streamlit.app/) 👷‍♀️
<br>
The purpose of this project is to increase workplace safety by creating a machine learning model that detects whether a worker is wearing a safety helmet or not.
<br>
This is a final project for Le Wagon Tokyo Data Science & AI batch #1639 and was presented on May 31st 2024 at the Shibuya Google for Startups campus.
<br>
[Presentation Video - Le Wagon Spring 2024 - Demo Day - Safety Helmet Detection](https://www.youtube.com/watch?v=CLhtyzqgObE) (From 29:45)
<br>
[Canva Presentation Link](https://helmet-detection.my.canva.site/)
<br>
The YOLO model was selected for it's excellent object detection capabilities and trained on custom data.
<br>
Specifically YOLO v8n was selected for its fast detection speed.
<br>
![image](https://github.com/Nsayre/helmet_det/assets/6730926/d4b6fad5-170d-4123-8be3-47d6f3e1f34c)
<br>
![image](https://github.com/Nsayre/helmet_det/assets/6730926/f48c1a9c-2ebd-491c-b639-5cffc8d23388)
## Getting Started
In order to perform training, raw data must be reformatted for the YOLO model. See preprocess.py for details.
### Installing Packages (Linux/WSL)
```
pip install -r requirements.txt
```
### Running Streamlit App locally (Linux/WSL)
```
streamlit run app/app.py
```
### Running Streamlit App in a Docker Container
```
docker build -t helmet_det_app .
docker run -p 8080:8080 helmet_det_app
```
Once the container is running, you can open the app in your web browser at localhost:8080
## Built With
- [YOLO](https://guides.rubyonrails.org/) - State of the art computer vision model
- [Streamlit](https://streamlit.io/) - Web app framework
## Acknowledgements
- [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) for training data
## Team Members
- [Nicholas Sayre](https://www.linkedin.com/in/nicholas-sayre/)
- [Karush Pradhan](https://www.linkedin.com/in/karushpradhan/)
- [Iliyas Kurmangali](https://www.linkedin.com/in/iliyas-kurmangali-432273188/)
- [Sayaka Tajima](https://www.linkedin.com/in/sayaka-tajima/)
## Next Steps
- Train model on expanded safety helmet data set.
- Potentially usable data #1: [PPE_detection](https://github.com/ZijianWang-ZW/PPE_detection/tree/master)
- Potentially usable data #2: [helmet-detection](https://github.com/wujixiu/helmet-detection)
- Include other safety equipment such as safety vests, gloves, or safety glasses
- Train YOLO v8 m or larger for better accuracy
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## License
This project is licensed under the MIT License
