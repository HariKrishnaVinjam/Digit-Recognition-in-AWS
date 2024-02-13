import sys
import os
import base64
import io
import boto3
import json
import csv
from PIL import Image, ImageOps
import requests
from io import BytesIO
import numpy as np

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT']

runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):


    data = json.loads(json.dumps(event))
    handWriEncImg = data["handWrittenEncodedImage"]
    hostImg = data["hostedImage"]
    
    if handWriEncImg != "empty":
        dec_img_str = base64.b64decode(str(handWriEncImg))
        img = Image.open(BytesIO(dec_img_str))
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = ImageOps.invert(ImageOps.grayscale(img))
        img = np.array(img.getdata(), dtype = np.uint8).reshape(28, 28)
    elif hostImg != "empty":
        response = requests.get(hostImg)
        img = Image.open(BytesIO(response.content))
        img = np.array(img.getdata(), dtype = np.uint8).reshape(img.size[0], img.size[1])
    
    hist = [0]*256
    rows = img.shape[0]
    cols = img.shape[1]
    
    for i in range(rows):
        for j in range(cols):
            hist[img[i][j]] += 1
    
    
    weighted_intra_class_variance = {}
    sigmaHi = 0
    probabilties = [0]*256
    
    for i in hist:
        sigmaHi += i
    for i in range(256):
        probabilties[i] = hist[i]/sigmaHi
    
    for t in range(0, 256):
        q1 = 0
        q2 = 0
        for i in range(t+1):
            q1 += probabilties[i]
        for j in range(t+1, 256):
            q2 += probabilties[j]
    
        m1 = 0
        m2 = 0
        for i in range(t+1):
            try:
                m1 += (i*probabilties[i])/q1
            except:
                continue
        for j in range(t+1, 256):
            try:
                m2 += (j*probabilties[j])/q2
            except:
                continue
    
        icv1 = 0
        icv2 = 0
        for i in range(t+1):
            try:
                icv1 += (((i-m1)**2)*probabilties[i])/q1
            except:
                continue
        for j in range(t+1, 256):
            try:
                icv2 += (((j-m2)**2)*probabilties[j])/q2
            except:
                continue
    
        wicv = q1*icv1 + q2*icv2
        weighted_intra_class_variance[t] = wicv
    
    min_wicv = sys.maxsize
    threshold = 256
    
    for i in weighted_intra_class_variance:
        if weighted_intra_class_variance[i] < min_wicv:
            min_wicv = weighted_intra_class_variance[i]
            threshold = i
    
    
    bin_img = img
    rows = bin_img.shape[0]
    cols = bin_img.shape[1]
    for i in range(rows):
        for j in range(cols):
            if bin_img[i][j] >= threshold:
                bin_img[i][j] = 255
            else:
                bin_img[i][j] = 0
                
    test_arr = ','.join([ str(bin_img[i,j]) for i in range(28) for j in range(28)])
    
    

    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='text/csv',
                                      Body=test_arr)

    result = json.loads(response['Body'].read().decode())

    return {
            'statusCode':200,
            'body': result['predictions']
        }
