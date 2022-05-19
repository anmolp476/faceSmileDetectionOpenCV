import cv2 as cv

groupImg = cv.imread('Photos/brady-bunch-3.jpg')
cv.imshow('Group Image', groupImg)

grayScale = cv.cvtColor(groupImg, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray Lady Image', grayScale)

#This is trained face detecting data
faceCasc = cv.CascadeClassifier('faceHaarCascades.xml')

#This is trained smile detecting data
smileCasc = cv.CascadeClassifier('smileHaarCascades.xml')

#Use the parameters to detect a face, and return a list of the coordinates of any detected face
faceRect = faceCasc.detectMultiScale(grayScale, scaleFactor=1.1,minNeighbors=5)

#Return a list of coordinates for this variable too
smileRect = smileCasc.detectMultiScale(grayScale, scaleFactor=1.1, minNeighbors=70)

print(f'No. of smiles is {len(smileRect)}s')

'''Since faceRect returns the rectangular coordinates of the face, we can
   use loop through the list filled with those coordinates and use them to 
   draw a rectangle over the detected face. The coordinates are the x and y, and 
   the dimensions of the rectangle are w and h. 
   Rmr that faceRect returns a list of the rectangular
   coordinates of the detected face. Same concept for smileRect.'''

for (x, y, width, height) in faceRect:
    cv.rectangle(groupImg, (x,y), (x+width, y+height), (255,0,0), thickness=3)

for (xPos, yPos, width, height) in smileRect:
    #Now create the rectangle in the original image using those coordinates, and colour it green
    cv.rectangle(groupImg, (xPos,yPos), (xPos+width, yPos+height), (0,255,0), thickness=2)

cv.imshow('Deteced Lady Image', groupImg)

cv.waitKey(0)