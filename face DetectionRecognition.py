import cv2
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import face_recognition  # Alternative library for recognition

class FaceDetectionSystem:
    def __init__(self):
        # Initialize Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize DNN face detector
        self.dnn_net = None
        self.load_dnn_model()
        
        # Face recognition components
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_encoder = LabelEncoder()
        self.known_faces = []
        self.known_names = []
        
    def load_dnn_model(self):
        """Load DNN model for face detection"""
        try:
            # Download these files from OpenCV's repository or use face_recognition library
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.dnn_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                print("DNN model loaded successfully")
            else:
                print("DNN model files not found. Using Haar Cascades only.")
        except Exception as e:
            print(f"Error loading DNN model: {e}")
    
    def detect_faces_haar(self, image, scale_factor=1.1, min_neighbors=5):
        """Detect faces using Haar Cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        return faces
    
    def detect_faces_dnn(self, image, confidence_threshold=0.5):
        """Detect faces using DNN"""
        if self.dnn_net is None:
            return []
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def detect_faces_face_recognition(self, image):
        """Detect faces using face_recognition library"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Convert to (x, y, w, h) format
        faces = []
        for (top, right, bottom, left) in face_locations:
            faces.append((left, top, right-left, bottom-top))
        
        return faces
    
    def prepare_training_data(self, data_folder):
        """Prepare training data from folder structure: data_folder/person_name/images"""
        faces = []
        labels = []
        
        data_path = Path(data_folder)
        for person_folder in data_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                
                for image_file in person_folder.glob("*.jpg"):
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue
                    
                    # Detect faces in the image
                    detected_faces = self.detect_faces_haar(image)
                    
                    for (x, y, w, h) in detected_faces:
                        face_roi = image[y:y+h, x:x+w]
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        resized_face = cv2.resize(gray_face, (100, 100))
                        
                        faces.append(resized_face)
                        labels.append(person_name)
        
        return faces, labels
    
    def train_recognizer(self, data_folder):
        """Train the face recognizer"""
        print("Preparing training data...")
        faces, labels = self.prepare_training_data(data_folder)
        
        if len(faces) == 0:
            print("No training data found!")
            return
        
        # Encode labels
        numeric_labels = self.label_encoder.fit_transform(labels)
        
        # Train the recognizer
        print("Training face recognizer...")
        self.face_recognizer.train(faces, np.array(numeric_labels))
        
        # Save the model
        self.face_recognizer.save("face_recognizer.yml")
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        print("Training completed!")
    
    def load_recognizer(self):
        """Load trained recognizer"""
        try:
            self.face_recognizer.read("face_recognizer.yml")
            with open("label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
            print("Recognizer loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading recognizer: {e}")
            return False
    
    def recognize_face(self, face_image, confidence_threshold=50):
        """Recognize a face"""
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (100, 100))
        
        label, confidence = self.face_recognizer.predict(resized_face)
        
        if confidence < confidence_threshold:
            name = self.label_encoder.inverse_transform([label])[0]
            return name, confidence
        else:
            return "Unknown", confidence
    
    def process_video(self, source=0, method="haar"):
        """Process video stream for face detection/recognition"""
        cap = cv2.VideoCapture(source)
        
        # Load recognizer if available
        recognizer_loaded = self.load_recognizer()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces based on selected method
            if method == "haar":
                faces = self.detect_faces_haar(frame)
            elif method == "dnn":
                faces = self.detect_faces_dnn(frame)
            elif method == "face_recognition":
                faces = self.detect_faces_face_recognition(frame)
            else:
                faces = self.detect_faces_haar(frame)
            
            # Draw rectangles and recognize faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                if recognizer_loaded:
                    face_roi = frame[y:y+h, x:x+w]
                    name, confidence = self.recognize_face(face_roi)
                    
                    # Display name and confidence
                    text = f"{name} ({confidence:.1f})"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Face Detection/Recognition', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path, method="haar", save_result=True):
        """Process a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print("Error loading image!")
            return
        
        # Load recognizer if available
        recognizer_loaded = self.load_recognizer()
        
        # Detect faces
        if method == "haar":
            faces = self.detect_faces_haar(image)
        elif method == "dnn":
            faces = self.detect_faces_dnn(image)
        elif method == "face_recognition":
            faces = self.detect_faces_face_recognition(image)
        else:
            faces = self.detect_faces_haar(image)
        
        print(f"Found {len(faces)} face(s)")
        
        # Draw rectangles and recognize faces
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if recognizer_loaded:
                face_roi = image[y:y+h, x:x+w]
                name, confidence = self.recognize_face(face_roi)
                
                # Display name and confidence
                text = f"{name} ({confidence:.1f})"
                cv2.putText(image, text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Face {i+1}: {name} (confidence: {confidence:.1f})")
        
        # Display result
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result if requested
        if save_result:
            output_path = f"result_{Path(image_path).name}"
            cv2.imwrite(output_path, image)
            print(f"Result saved as {output_path}")


def main():
    """Main function to demonstrate usage"""
    detector = FaceDetectionSystem()
    
    print("Face Detection/Recognition System")
    print("1. Train recognizer with data")
    print("2. Process webcam (Haar)")
    print("3. Process webcam (DNN)")
    print("4. Process webcam (face_recognition)")
    print("5. Process image")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            data_folder = input("Enter path to training data folder: ")
            if os.path.exists(data_folder):
                detector.train_recognizer(data_folder)
            else:
                print("Data folder not found!")
        
        elif choice == "2":
            print("Starting webcam with Haar Cascades (Press 'q' to quit)")
            detector.process_video(method="haar")
        
        elif choice == "3":
            print("Starting webcam with DNN (Press 'q' to quit)")
            detector.process_video(method="dnn")
        
        elif choice == "4":
            print("Starting webcam with face_recognition (Press 'q' to quit)")
            detector.process_video(method="face_recognition")
        
        elif choice == "5":
            image_path = input("Enter path to image: ")
            if os.path.exists(image_path):
                method = input("Choose method (haar/dnn/face_recognition): ").lower()
                if method not in ["haar", "dnn", "face_recognition"]:
                    method = "haar"
                detector.process_image(image_path, method=method)
            else:
                print("Image file not found!")
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()