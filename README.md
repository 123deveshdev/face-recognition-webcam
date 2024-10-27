# A simple corrected example fron webcam example by ageitgey on "https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py"

The correction is on line 51 from "rgb_small_frame = small_frame[:, :, ::-1]" to "rgb_small_img = np.ascontiguousarray(small_img[:, :, ::-1])"

NOTE:-If a trouble in installing dlib install wheels from "https://github.com/z-mahmud22/Dlib_Windows_Python3.x"
