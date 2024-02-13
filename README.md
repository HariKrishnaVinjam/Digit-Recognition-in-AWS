1. To succesfully run this project all you need to do is open the file named "API_LINK.txt"
2. There you can find the API Link.
3. Next, go to postman.com and select HTTP request with the post method and paste the API link.
4. Next, go back to the github repository and open the folder named "test_cases".
5. There you can find all the test cases for both hand written images and hosted images(On kaggle website). Open each test case to find the body for POST request.
6. Next, go back to POSTMAN and under the body tab place this body text.
7. Now send this post request and you will get the predicted label output


NOTE: 

A) All hand written images are encoded to base64 string and then they are sent to the lambda function where it will be decode into an image.


B) All test cases predict correct labels except the following test cases:

  1.test_case_hand_Written_Encoded_Image_7

  2.test_case_hand_Written_Encoded_Image_8

  3.test_case_hand_Written_Encoded_Image_9
