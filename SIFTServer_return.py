
# coding: utf-8

# In[1]:


#!/usr/bin/env python
 
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np
import sys
import time
import requests


# In[2]:


def process_img(payload):
    result = []
    t0 = time.process_time()
    
    search_params = dict(checks = 20) # this is for the flann-based matcher
    MIN_MATCH_COUNT = 300 # very relaxed matching at 10 matches minimum
    detector = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    ref = "train.jpg"
 
    query_img = payload
    ref_img = cv2.imread(ref, 0)

    rKP, rDES = detector.detectAndCompute(ref_img, None)        
    qKP, qDES = detector.detectAndCompute(query_img, None)

    try:
        matches = matcher.knnMatch(rDES,qDES,k=2)
    except:
        state = "Matching Error: not enough query points."
        result.append(state)
        result.append(query_img)
        t1 = time.process_time()
        print(state,"Time to process:", t1-t0)
        result.append((t1-t0))
        return result    

    # store all the good matches as per Lowe's ratio test.
    good = []
    distances = []

    for m,n in matches:
        distances.append(m.distance)
        if m.distance < 0.75*n.distance:
            good.append(m)

    good = sorted(good, key = lambda x:x.distance)
    
    if len(good)>MIN_MATCH_COUNT:
        state = "Enough matches: object is propbably in view."
        # extract location of points in both images
        src_pts = np.float32([ rKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ qKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # find the perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        # get the transform points in the (captured) query image
        h,w = ref_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        try:
            dst = cv2.perspectiveTransform(pts,M)
            # draw the transformed image
            result.append(state)
            result.append(cv2.drawContours(query_img,[np.int32(dst)],-1,(255,0,0),6))
            result.append(dst)
        except:
            state = "Error getting perspective transform."
            result.append(state)
            result.append(query_img)
        finally:
            t1 = time.process_time()
            print(state,"Time to process:", t1-t0)
            result.append((t1-t0))

    else:
        state = "Not enough matches: object is not in view or is not clear."
        result.append(state)
        result.append(query_img)
        t1 = time.process_time()
        print(state,"Time to process:", t1-t0)
        result.append((t1-t0))

    return result


# In[3]:


# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):

  # GET
    def do_GET(self):
        # Send response status code
        self.send_response(200)

        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()

        # Send message back to client
        message = "Hello world!"
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return
    
    
    def do_POST(self):

        print( "incoming http: ", self.path )

        # Gets the size of data
        content_length = int(self.headers['Content-Length'])
        content_type = self.headers['Content-type']
        # Gets the data itself. Also, we over-catch by 16 bytes.
        
        self.send_response(200)
        self.close_connection
        
        if "image" in content_type:
            post_data = self.rfile.read(content_length+16)

            #print("Length of content:", len(post_data))
            #print("Before pruning", post_data[:32])

            # We remove the first set of bytes up until the first
            # carraige return and newline.
            for b in np.arange(len(post_data)):
                # Checking where the first newline is
                if post_data[b] == 13:
                    post_data = post_data[b+2:]
                    break
                else:
                    continue

            #stream = io.BytesIO(post_data)
            tmp = 'tmp.jpg'
            with open('tmp.jpg','wb') as out:
                out.write(post_data)

            out.close()
            # Converting the byte buffer to an numpy array for opencv
            nparr = np.fromstring(post_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            final_img = process_img(img_np)
            
            output = 'res.jpg'
            cv2.imwrite(output, final_img[1])
            
            #test_payload = {'file' : open('res.jpg','rb')}
            #rPost = requests.post('http://192.168.43.205:8081', files = test_payload)
            
            # Send headers
            self.send_header('Content-type','text/html')
            #self.send_header('Content-type','image/jpeg')
            
            # Send message back to client
            # Write content as utf-8 data
            print(len(final_img))
            if len(final_img)>3:
                message = bytes(str(final_img[3])+ "\n" + np.array2string(final_img[2],precision=0,separator=','), "utf8")
                print(str(message))
                self.send_header('Content-length',str(len(message)))
                self.end_headers()
                
                self.wfile.write(message)
            else:
                message = bytes(str(final_img[2])+ "\n" + final_img[0], "utf8")
                self.send_header('Content-length',str(len(message)))
                self.end_headers()
                
                self.wfile.write(message)
            #self.wfile.write(res.read())

        else:
            print("Content-type is ", content_type,". Should be image.")
            self.send_header('Content-type','text/html')
            self.end_headers()

            # Send message back to client
            message = "Object not detected."
            # Write content as utf-8 data
            self.wfile.write(bytes(message, "utf8"))

        return
        #client.close()


# In[4]:


def run():
    print('starting server...')
 
  # Server settings
  # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('0.0.0.0', 8081)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()



# In[5]:


run()

