from PIL import ExifTags,Image
import io
import requests
from flask import Flask, render_template, request, jsonify,flash,redirect,url_for,session
import sqlite3
import os
import psutil
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import torch
import gc



app = Flask(__name__)
app.secret_key="4TuBbgTs1T8ILHrNOcacw8eLwXZhsaQP"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
error_flag= {"error_detected":False}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
model1 = YOLO('satellite.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)



#allowextension function
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


GPSINFO_TAG = next(
    tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"
)

device = next(model.model.parameters()).device
print(f"Model is running on: {device}")


#image coordinates cal
def decimal_coords(coords, ref):
    decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
    if ref == "S" or ref =='W' :
        decimal_degrees = -1 * decimal_degrees
    return decimal_degrees


def check_memory():
    process=psutil.Process(os.getpid())
    memory_usage=process.memory_info().rss / (1024 * 1024)
    if memory_usage > 390:
        error_flag["error_detected"]=True


@app.route("/error-status",methods=['GET'])
def error_status():
    check_memory()
    if error_flag['error_detected']:
        error_flag['error_detected']=False
        
        return jsonify({"error":True,"message":"Try again"}), 500
    return render_template('index.html') 

@app.route('/')
def index():
    return render_template('index.html')

#signup function
@app.route('/signup',methods=['POST','GET'])
def signup():
    if request.method=='POST':
        try:
            email=request.form['email']
            password=request.form['password']
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute("insert into register(email,password)values(?,?)",(email,password))
            conn.commit()
            flash("Signup successfully","success_signup")
        except:
            flash("Account already exists","error")
        finally:
            conn.close()
            return redirect('login')  
    return render_template("index.html")

#login function
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        conn = sqlite3.connect('database.db')
        conn.row_factory=sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("select * from register where email=? and password=? ",(email,password))
        data=cursor.fetchone()
        if data:
            session["email"]=data["email"]
            session["password"]=data["password"]
            flash("login successfully","success")
            return redirect('/')
        else:
            flash("Incorrect email or password","danger")
            return redirect('login')
    return render_template("index.html")

#logout function
@app.route("/logout",methods=['POST','GET'])
def logout():
    if request.method=='POST':
        return redirect('/')
    return render_template('index.html')

        

#satellite image function
@app.route('/satellite-detect', methods=['POST'])
def detect_objects():
    data = request.get_json()
    north, south, east, west = data['north'], data['south'], data['east'], data['west']
    zoom=data['zoom']
    
   
    image = fetch_satellite_image(north, south, east, west,zoom)
    
 
    detection_results = perform_detection(image, data)
    
    return jsonify(detection_results)

def fetch_satellite_image(north, south, east, west,zoom):
   

    url = f"https://maps.gomaps.pro/maps/api/staticmap?center={(north+south)/2},{(east+west)/2}&zoom={zoom}&size=928x544&maptype=satellite&key=AlzaSy2eGajaEJKBQ6RMeaASWbYxGv4RpkDDdIp"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Failed to fetch satellite image")

def perform_detection(image_bytes, bounds):
    try:
        image = Image.open(io.BytesIO(image_bytes))

        results = model1(image,imgsz=928,conf=0.35)

        north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
        img_width,img_height=image.size

        detected_objects = []
        for result in results:
            for obj in result.boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_idx = obj
                confidence=confidence.item()
                label = model1.names[int(class_idx)]
            
                rel_x_min, rel_x_max = x_min.item() / img_width, x_max.item() / img_width
                rel_y_min, rel_y_max = y_min.item() / img_height, y_max.item() / img_height

        
                 obj_north = north + rel_y_min * (south-north)
                 obj_south = north + rel_y_max * (south -north)
                 obj_west = west + rel_x_min * (east - west)
                 obj_east = west + rel_x_max * (east - west)

       
                 obj_lat = (obj_north + obj_south) / 2
                 obj_lng = (obj_east + obj_west) / 2

                detected_objects.append({
                   'name': label,
                   'type': label,
                   'conf': confidence,
                  'north': obj_north,
                  'south': obj_south,
                   'east': obj_east,
                   'west': obj_west,
                    'lat': obj_lat,
                    'lng': obj_lng
               })

        return {'objects': detected_objects}
    except Exception as e:
        print(f"Error: {e}")
    finally:
        del results
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        check_memory() 

#360 image function
@app.route('/360-detect', methods=['POST'])
def save_image():
    data = request.json
    lat = data['lat']
    lng = data['lng']
    pitch = data['pitch']
    heading = data['heading']  
    fov = data['fov']
    zoom = data['zoom']

    street_view_url = f"https://maps.gomaps.pro/maps/api/streetview?size=800x600&location={lat},{lng}&pitch={pitch}&heading={heading}&fov={fov}&zoom={zoom}&key=AlzaSy0AkjtVg-gRrHMD7pDGLISDlkyVxSGHiJb"

    try:
        response = requests.get(street_view_url)
        if response.status_code != 200:
            return jsonify({"message": "No imagery available for this location", "error": response.text}), 404

        file_path = f'static/streetview.jpg'
        with open(file_path, 'wb') as f:
            f.write(response.content)
        results=model1('static/streetview.jpg',save=True,project='static',exist_ok=True,conf=0.30)

        return jsonify({"message": "Image saved successfully", "file_path": file_path})
    except Exception as e:
        return jsonify({"message": "Error fetching image", "error": str(e)}), 500
    

#upload function
@app.route('/upload',methods=['POST','GET'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({"message": "please upload the imagery", "error": "file"}), 404
        file = request.files['file']
        detect_coords=[]   
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            image=Image.open('static/uploads/'+filename)
            if image.getexif():
                try:                
                    info = image.getexif()
                    gpsinfo = info.get_ifd(GPSINFO_TAG)
                    obj_lat=(decimal_coords(gpsinfo[2], gpsinfo[1]))
                    obj_lng=(decimal_coords(gpsinfo[4], gpsinfo[3]))
                    detect_coords.append({
                        'lat':obj_lat,
                        'lng':obj_lng
                    })
                    detect_coords= {'objects':detect_coords,'message':"uploaded imagery shown on Google Map"}
                    return  jsonify(detect_coords)
                except AttributeError:
                    return jsonify({"message": "No EXIF data found", "error": "No EXIF data"}), 404
            else:
                return jsonify({"message": "No exif data found! Try manual input", "error": "no EXIF data"}), 404
            
    return render_template('index.html')



if __name__ == '__main__':
    app.run()
