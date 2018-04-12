/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package opencvimagesegmentation;

import java.io.ByteArrayInputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author asmin
 */
public class FXMLDocumentController implements Initializable {

    private Label label;
    @FXML
    private CheckBox canny;
    @FXML
    private Slider threshold;
    @FXML
    private CheckBox dilateErode;
    @FXML
    private CheckBox inverse;
    @FXML
    private Button cameraButton;
    @FXML
    private Insets x1;
    @FXML
    private ImageView originalFrame;
    private Mat frame;

    // a timer for acquiring the video stream
    private ScheduledExecutorService timer;
    // the OpenCV object that performs the video capture
    private VideoCapture capture = new VideoCapture();
    // a flag to change the button behavior
    private boolean cameraActive = false;
    Point clickedPoint = new Point(0, 0);
    private Mat oldFrame;

    private void handleButtonAction(ActionEvent event) {
        System.out.println("You clicked me!");
        label.setText("Hello World!");
    }

    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }

    @FXML
    private void cannySelected(ActionEvent event) {
        // check whether the other checkbox is selected and deselect it
        if (this.dilateErode.isSelected()) {
            this.dilateErode.setSelected(false);
            this.inverse.setDisable(true);
        }

        // enable the threshold slider
        if (this.canny.isSelected()) {
            this.threshold.setDisable(false);
        } else {
            this.threshold.setDisable(true);
        }

        // now the capture can start
        this.cameraButton.setDisable(false);
    }

    @FXML
    private void dilateErodeSelected(ActionEvent event) {

        // check whether the canny checkbox is selected, deselect it and disable
        // its slider
        if (this.canny.isSelected()) {
            this.canny.setSelected(false);
            this.threshold.setDisable(true);
        }

        if (this.dilateErode.isSelected()) {
            this.inverse.setDisable(false);
        } else {
            this.inverse.setDisable(true);
        }

        // now the capture can start
        this.cameraButton.setDisable(false);
    }

    @FXML
    private void startCamera(ActionEvent event) {
        if (!this.cameraActive) {
            // disable setting checkboxes
            this.canny.setDisable(true);
            this.dilateErode.setDisable(true);

            // start the video capture
            this.capture.open(0);

            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = new Runnable() {

                    @Override
                    public void run() {
                        // effectively grab and process a single frame
                        Mat frame = new Mat();
                        capture.read(frame);
                        if (!frame.empty()) {
                            // handle edge detection
                            if (canny.isSelected()) {
                                frame = doCanny(frame);
                                //frame = this.doSobel(frame);
                            } // foreground detection
                            else if (dilateErode.isSelected()) {
                                // Es. 2.1
//                                 frame = doBackgroundRemovalFloodFill(frame);
                                // Es. 2.2
//                                frame = doBackgroundRemovalAbsDiff(frame);
                                // Es. 2.3
                                frame = doBackgroundRemoval(frame);

                            }

                        }
                        // convert and show the frame
                        Image imageToShow = mat2Image(frame);
                        Platform.runLater(new Runnable() {
                            @Override
                            public void run() {
                                originalFrame.setImage(imageToShow);

                            }
                        });
                    }
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Start Camera");
            // enable setting checkboxes
            this.canny.setDisable(false);
            this.dilateErode.setDisable(false);

            // stop the timer
            this.timer.shutdown();
            this.capture.release();
        }
    }

    private Mat doCanny(Mat frame) {
        Mat grayImage = new Mat();
        Mat detectedEdges = new Mat();
        Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

        Imgproc.Canny(detectedEdges, detectedEdges, this.threshold.getValue(),
                this.threshold.getValue() * 3, 3, false);

        Mat dest = new Mat();
        Core.add(dest, Scalar.all(0), dest);

        frame.copyTo(dest, detectedEdges);

        return dest;

    }

    private Mat doBackgroundRemoval(Mat frame) {
        Mat hsvImg = new Mat();
        List<Mat> hsvPlanes = new ArrayList<>();
        hsvImg.create(frame.size(), CvType.CV_8U);
        Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
        //Now let's split the three channels of the image:
        Core.split(hsvImg, hsvPlanes);

        List<Mat> hue = new ArrayList<>();
        hue.add(hsvPlanes.get(0));
        Mat hist_hue = new Mat();
        MatOfInt histSize = new MatOfInt(180);
        Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

        double average = 0;
        for (int h = 0; h < 180; h++) {
            average += (hist_hue.get(h, 0)[0] * h);
        }
        average = average / hsvImg.size().height / hsvImg.size().width;

        Mat thresholdImg = new Mat();
        double threshValue = this.threshold.getValue();
        if (this.inverse.isSelected()) {
            Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY_INV);
        } else {
            Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);
        }

        Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

        Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

        Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);
        Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        frame.copyTo(foreground, thresholdImg);

        return foreground;
    }

    private Mat doBackgroundRemovalAbsDiff(Mat currFrame) {
        Mat greyImage = new Mat();
        Mat foregroundImage = new Mat();

        if (oldFrame == null) {
            oldFrame = currFrame;
        }

        Core.absdiff(currFrame, oldFrame, foregroundImage);
        Imgproc.cvtColor(foregroundImage, greyImage, Imgproc.COLOR_BGR2GRAY);

        int thresh_type = Imgproc.THRESH_BINARY_INV;
        if (this.inverse.isSelected()) {
            thresh_type = Imgproc.THRESH_BINARY;
        }

        Imgproc.threshold(greyImage, greyImage, 10, 255, thresh_type);
        currFrame.copyTo(foregroundImage, greyImage);

        oldFrame = currFrame;
        return foregroundImage;

    }

    private Mat doBackgroundRemovalFloodFill(Mat frame) {

        Scalar newVal = new Scalar(255, 255, 255);
        Scalar loDiff = new Scalar(50, 50, 50);
        Scalar upDiff = new Scalar(50, 50, 50);
        Point seedPoint = clickedPoint;
        Mat mask = new Mat();
        Rect rect = new Rect();

        // Imgproc.floodFill(frame, mask, seedPoint, newVal);
        Imgproc.floodFill(frame, mask, seedPoint, newVal, rect, loDiff, upDiff, Imgproc.FLOODFILL_FIXED_RANGE);

        return frame;
    }

    private Image mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);
        Image imageToShow = new Image(new ByteArrayInputStream(buffer.toArray()));
        return imageToShow;
    }
}
