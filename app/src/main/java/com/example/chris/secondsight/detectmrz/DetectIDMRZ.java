package com.example.chris.secondsight.detectmrz;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.os.Environment;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

//import java.awt.FlowLayout;
//import java.awt.Image;
//import java.awt.image.BufferedImage;
//import java.awt.image.DataBufferByte;
//import java.awt.image.RenderedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

//import javax.imageio.ImageIO;
//import javax.swing.ImageIcon;
//import javax.swing.JFrame;
//import javax.swing.JLabel;

//import cern.colt.*;
//import cern.colt.matrix.DoubleMatrix2D;
//import cern.colt.matrix.linalg.Matrix2DMatrix2DFunction;

import com.example.chris.secondsight.CameraActivity;
import com.example.chris.secondsight.R;
import com.googlecode.tesseract.android.*;

/**
 * Created by chris on 8/16/16.
 */
public class DetectIDMRZ {
    Context thisContext;

    Activity thisActivity;

    // The reference image (this detector's target).
    private final Mat mReferenceImage;

    // Features of the reference image.
    private final MatOfKeyPoint mReferenceKeypoints = new MatOfKeyPoint();

    // Descriptors of the reference image's features.
    private final Mat mReferenceDescriptors = new Mat();

    // The corner coordinates of the reference image, in pixels.
    // CvType defines the color depth, number of channels, and
    // channel layout in the image. Here, each point is represented
    // by two 32-bit floats.
    private final Mat mReferenceCorners = new Mat(4, 1, CvType.CV_32FC2);

    // Features of the scene (the current frame).
    private final MatOfKeyPoint mSceneKeypoints = new MatOfKeyPoint();

    // Descriptors of the scene's features.
    private final Mat mSceneDescriptors = new Mat();

    // Tentative corner coordinates detected in the scene, in pixels.
    private final Mat mCandidateSceneCorners = new Mat(4, 1, CvType.CV_32FC2);

    // Good corner coordinates detected in the scene, in pixels.
    private final Mat mSceneCorners = new Mat(0, 0, CvType.CV_32FC2);

    // The good detected corner coordinates, in pixels, as integers.
    private final MatOfPoint mIntSceneCorners = new MatOfPoint();

    // A grayscale version of the scene.
    private final Mat mGraySrc = new Mat();

    // Tentative matches of scene features and reference features.
    private final MatOfDMatch mMatches = new MatOfDMatch();

    // A feature detector, which finds features in images.
    private final FeatureDetector mFeatureDetector = FeatureDetector.create(FeatureDetector.ORB);

    // A descriptor extractor, which creates descriptors of features.
    private final DescriptorExtractor mDescriptorExtractor =
            DescriptorExtractor.create(DescriptorExtractor.ORB);

    // A descriptor matcher, which matches features based on their descriptors.
    private final DescriptorMatcher mDescriptorMatcher =
            DescriptorMatcher.create(
                    DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);

    // The color of the outline drawn around the detected image.
    private final Scalar mLineColor = new Scalar(0, 255, 0);

    public static final String PACKAGE_NAME = "com.example.chris.secondsight.detectmrz";
    public static final String DATA_PATH = Environment
            .getExternalStorageDirectory().toString() + "/SecondSight/";

    // You should have the trained data file in assets folder
    // You can get them at:
    // http://code.google.com/p/tesseract-ocr/downloads/list
    public static final String lang = "eng";

    private static final String TAG = "SecondSight.java";

    //Methods
    public Mat resize(Mat img, int width, int height, int inter){

        inter = Imgproc.INTER_AREA;

        Size imgDim = img.size();

        Size dim = null;

        double r = 1;

        if(width <= 0 && height <= 0)
            return img;

        if (height == 0)
        {
            r =  width/imgDim.width;
            dim = new Size(width, (int)(img.height() * r));
        }
        else if(width == 0)
        {
            r = height/imgDim.height;
            dim = new Size((int)(img.width() * r), height);
        }
        else if (width > 0 && height > 0)
        {
            dim = new Size(width, height);
        }


        //resize the image
        Mat resized = new Mat();

        Imgproc.resize(img, resized, dim, 0, 0, inter);


        return resized;
    }

/*    public void displayImage(Image img2, String label)
    {
        //BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
        ImageIcon icon=new ImageIcon(img2);

        JFrame frame=new JFrame(label);

        frame.setLayout(new FlowLayout());

        frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);

        JLabel lbl=new JLabel();

        lbl.setIcon(icon);

        frame.add(lbl);

        frame.setVisible(true);

        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
    }

    public Image toBufferedImage(Mat m){
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            Mat m2 = new Mat();
            Imgproc.cvtColor(m,m2,Imgproc.COLOR_BGR2RGB);
            type = BufferedImage.TYPE_3BYTE_BGR;
            m = m2;
        }
        byte [] b = new byte[m.channels()*m.cols()*m.rows()];
        m.get(0,0,b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        image.getRaster().setDataElements(0, 0, m.cols(),m.rows(), b);
        return image;

    }
*/
    public DetectIDMRZ(final Context context,
                       final Activity activity,
                       final int referenceImageResourceID) throws IOException {
        //Constructor

        thisContext = context;
        thisActivity = activity;

        // Load the reference image from the app's resources.
        // It is loaded in BGR (blue, green, red) format.
        mReferenceImage = Utils.loadResource(context, referenceImageResourceID, Imgcodecs.CV_LOAD_IMAGE_COLOR);

        // Create grayscale and RGBA versions of the reference image.
        final Mat referenceImageGray = new Mat();
        Imgproc.cvtColor(mReferenceImage, referenceImageGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(mReferenceImage, mReferenceImage, Imgproc.COLOR_BGR2RGBA);

        // Store the reference image's corner coordinates, in pixels.
        mReferenceCorners.put(0, 0, new double[]{0.0, 0.0});
        mReferenceCorners.put(1, 0, new double[]{referenceImageGray.cols(), 0.0});
        mReferenceCorners.put(2, 0, new double[]{referenceImageGray.cols(), referenceImageGray.rows()});
        mReferenceCorners.put(3, 0, new double[]{0.0, referenceImageGray.rows()});

        // Detect the reference features and compute their
        // descriptors.
        mFeatureDetector.detect(referenceImageGray, mReferenceKeypoints);

        mDescriptorExtractor.compute(referenceImageGray, mReferenceKeypoints, mReferenceDescriptors);

        String[] paths = new String[]{DATA_PATH, DATA_PATH + "tessdata/"};

        for (String path : paths) {
            File dir = new File(path);
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    Log.v(TAG, "ERROR: Creation of directory " + path + " on sdcard failed");
                    return;
                } else {
                    Log.v(TAG, "Created directory " + path + " on sdcard");
                }
            }

        }

        // lang.traineddata file with the app (in assets folder)
        // You can get them at:
        // http://code.google.com/p/tesseract-ocr/downloads/list
        // This area needs work and optimization
        if (!(new File(DATA_PATH + "tessdata/" + lang + ".traineddata")).exists()) {
            try {

                AssetManager assetManager = context.getAssets();
                InputStream in = assetManager.open("tessdata/" + lang + ".traineddata");
                //GZIPInputStream gin = new GZIPInputStream(in);
                OutputStream out = new FileOutputStream(DATA_PATH
                        + "tessdata/" + lang + ".traineddata");

                // Transfer bytes from in to out
                byte[] buf = new byte[1024];
                int len;
                //while ((lenf = gin.read(buff)) > 0) {
                while ((len = in.read(buf)) > 0) {
                    out.write(buf, 0, len);
                }
                in.close();
                //gin.close();
                out.close();

                Log.v(TAG, "Copied " + lang + " traineddata");
            } catch (IOException e) {
                Log.e(TAG, "Was unable to copy " + lang + " traineddata " + e.toString());
            }
        }
    }

    private void findSceneCorners() {

        final List<DMatch> matchesList = mMatches.toList();
        if (matchesList.size() < 4) {
            // There are too few matches to find the homography.
            return;
        }

        final List<KeyPoint> referenceKeypointsList = mReferenceKeypoints.toList();

        final List<KeyPoint> sceneKeypointsList = mSceneKeypoints.toList();

        // Calculate the max and min distances between keypoints.
        double maxDist = 0.0;

        double minDist = Double.MAX_VALUE;

        for (final DMatch match : matchesList) {
            final double dist = match.distance;
            if (dist < minDist) {
                minDist = dist;
            }
            if (dist > maxDist) {
                maxDist = dist;
            }
        }

        // The thresholds for minDist are chosen subjectively
        // based on testing. The unit is not related to pixel
        // distances; it is related to the number of failed tests
        // for similarity between the matched descriptors.
        if (minDist > 50.0) {
            // The target is completely lost.
            // Discard any previously found corners.
            mSceneCorners.create(0, 0, mSceneCorners.type());
            return;
        } else if (minDist > 25.0) {
            // The target is lost but maybe it is still close.
            // Keep any previously found corners.
            return;
        }

        // Identify "good" keypoints based on match distance.
        final ArrayList<Point> goodReferencePointsList = new ArrayList<Point>();
        final ArrayList<Point> goodScenePointsList = new ArrayList<Point>();
        final double maxGoodMatchDist = 1.75 * minDist;

        for (final DMatch match : matchesList) {
            if (match.distance < maxGoodMatchDist) {
                goodReferencePointsList.add(
                        referenceKeypointsList.get(match.trainIdx).pt);
                goodScenePointsList.add(
                        sceneKeypointsList.get(match.queryIdx).pt);
            }
        }

        if (goodReferencePointsList.size() < 4 ||
                goodScenePointsList.size() < 4) {
            // There are too few good points to find the homography.
            return;
        }

        // There are enough good points to find the homography.
        // (Otherwise, the method would have already returned.)

        // Convert the matched points to MatOfPoint2f format, as
        // required by the Calib3d.findHomography function.
        final MatOfPoint2f goodReferencePoints = new MatOfPoint2f();
        goodReferencePoints.fromList(goodReferencePointsList);
        final MatOfPoint2f goodScenePoints = new MatOfPoint2f();
        goodScenePoints.fromList(goodScenePointsList);

        // Find the homography.
        final Mat homography = Calib3d.findHomography(
                goodReferencePoints, goodScenePoints);

        // Use the homography to project the reference corner
        // coordinates into scene coordinates.
        Core.perspectiveTransform(mReferenceCorners,
                mCandidateSceneCorners, homography);

        // Convert the scene corners to integer format, as required
        // by the Imgproc.isContourConvex function.
        mCandidateSceneCorners.convertTo(mIntSceneCorners,
                CvType.CV_32S);

        // Check whether the corners form a convex polygon. If not,
        // (that is, if the corners form a concave polygon), the
        // detection result is invalid because no real perspective can
        // make the corners of a rectangular image look like a concave
        // polygon!
        if (Imgproc.isContourConvex(mIntSceneCorners)) {
            // The corners form a convex polygon, so record them as
            // valid scene corners.
            mCandidateSceneCorners.copyTo(mSceneCorners);
        }
    }

    public void detectID(Mat img)
    {
        //displayImage(toBufferedImage(img), "orig image before scale");


//        if (img.width() > 800)
//            while (img.width() > 800)// || img.height() > 600)
//                Imgproc.pyrDown(img, img);

        //displayImage(toBufferedImage(img), "orig image");

        //
        //START FIND REFERENCE IMAGE
        //

        // Convert the scene to grayscale.
        Imgproc.cvtColor(img, mGraySrc, Imgproc.COLOR_RGBA2GRAY);

        // Detect the scene features, compute their descriptors,
        // and match the scene descriptors to reference descriptors.
        mFeatureDetector.detect(mGraySrc, mSceneKeypoints);

        mDescriptorExtractor.compute(mGraySrc, mSceneKeypoints, mSceneDescriptors);

        mDescriptorMatcher.match(mSceneDescriptors, mReferenceDescriptors, mMatches);

        // Attempt to find the target image's corners in the scene.
        findSceneCorners();

        // If the corners have been found, draw an outline around the
        // target image.
        // Else, draw a thumbnail of the target image.
        Mat foundIDImg = new Mat();
        draw(img, img);

        //Draw matches
        Mat referenceImageGray = new Mat();

        Imgproc.cvtColor(mReferenceImage, referenceImageGray, Imgproc.COLOR_BGR2GRAY);
        Mat img3 = new Mat();
        Features2d.drawMatches(mGraySrc,mSceneKeypoints,referenceImageGray,mReferenceKeypoints,
                mMatches, img3);

        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES).toString()+"/SecondSight/matches.png", img3);

        //displayImage(toBufferedImage(foundIDImg), "found id?");

        //
        //END FIND REFERENCE IMAGE
        //
    }

    public Mat detectMRZ(Mat img)
    {
        //Mat img = Imgcodecs.imread(photoPath);

        Mat roi = new Mat();

        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13,5));

        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,21));

        if (img.width() > 800)
            // load the image, resize it, and convert it to grayscale
            img = resize(img, 800, 600, Imgproc.INTER_AREA);

        //displayImage(toBufferedImage(img), "orig image resized");

        Mat gray = new Mat();

        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        //displayImage(toBufferedImage(gray), "image in grayscale");

        //smooth the image using a 3x3 Gaussian, then apply the blackhat
        //morphological operator to find dark regions on a light background
        Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);

        //displayImage(toBufferedImage(gray), "gaussian blur");

        Mat blackhat = new Mat();
        Imgproc.morphologyEx(gray, blackhat, Imgproc.MORPH_BLACKHAT, rectKernel);

        //displayImage(toBufferedImage(blackhat), "blackhat");

        //compute the Scharr gradient of the blackhat image and scale the
        //result into the range [0, 255]
        Mat gradX = new Mat();
        //gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        Imgproc.Sobel(blackhat, gradX, CvType.CV_32F, 1, 0, -1, 1, 0);
        //gradX = Matrix absolute(gradX)

        //displayImage(toBufferedImage(gradX), "sobel");

        //(minVal, maxVal) = (np.min(gradX), np.max(gradX))
        MinMaxLocResult minMaxVal = Core.minMaxLoc(gradX);

        //gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX.convertTo(gradX,CvType.CV_8U,255.0/(minMaxVal.maxVal-minMaxVal.minVal),-255.0/minMaxVal.minVal);

        //displayImage(toBufferedImage(gradX), "sobel converted to CV_8U");

        //apply a closing operation using the rectangular kernel to close
        //gaps in between letters -- then apply Otsu's thresholding method
        Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, rectKernel);

        //displayImage(toBufferedImage(gradX), "closing operation morphology");

        Mat thresh = new Mat();
        Imgproc.threshold(gradX, thresh, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        //displayImage(toBufferedImage(thresh), "applied threshold");

        // perform another closing operation, this time using the square
        // kernel to close gaps between lines of the MRZ, then perform a
        // series of erosions to break apart connected components
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, sqKernel);

        //displayImage(toBufferedImage(thresh), "another closing operation morphology");

        Imgproc.erode(thresh, thresh, new Mat(), new Point(-1,-1), 4);

        //displayImage(toBufferedImage(thresh), "erode");
        // during thresholding, it's possible that border pixels were
        // included in the thresholding, so let's set 5% of the left and
        // right borders to zero
        int pRows = (int)(img.rows() * 0.05);
        int pCols = (int)(img.cols() * 0.05);

        //thresh[:, 0:pCols] = 0;
        //thresh.put(thresh.rows(), pCols, 0);
        //thresh[:, image.cols() - pCols] = 0;
        for (int i=0; i <= thresh.rows(); i++)
            for (int j=0; j<=pCols; j++)
                thresh.put(i, j, 0);

        //thresh[:, image.cols() - pCols] = 0;
        for (int i=0; i <= thresh.rows(); i++)
            for (int j=img.cols()-pCols; j<=img.cols(); j++)
                thresh.put(i, j, 0);

        //displayImage(toBufferedImage(thresh), "");

        // find contours in the thresholded image and sort them by their
        // size
        List<MatOfPoint> cnts = new ArrayList<MatOfPoint>();

        Imgproc.findContours(thresh.clone(), cnts, new Mat(), Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);

        //cnts.sort(Imgproc.contourArea(contour));//, Imgproc.contourArea(cnts, true))

        // loop over the contours
        for (MatOfPoint c : cnts){
            // compute the bounding box of the contour and use the contour to
            // compute the aspect ratio and coverage ratio of the bounding box
            // width to the width of the image
            Rect bRect = Imgproc.boundingRect(c);
            int x=bRect.x;
            int y=bRect.y;
            int w=bRect.width;
            int h=bRect.height;

            int grWidth = gray.width();

            float ar = (float)w / (float)h;
            float crWidth = (float)w / (float)grWidth;

            // check to see if the aspect ratio and coverage width are within
            // acceptable criteria
            if (ar > 4 && crWidth > 0.75){
                // pad the bounding box since we applied erosions and now need
                // to re-grow it
                int pX = (int)((x + w) * 0.03);
                int pY = (int)((y + h) * 0.03);
                x = x - pX;
                y = y - pY;
                w = w + (pX * 2);
                h = h + (pY * 2);

                // extract the ROI from the image and draw a bounding box
                // surrounding the MRZ

                roi = new Mat(img, new Rect(x, y, w, h));

                Imgproc.rectangle(img, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 255, 0), 2);


                //displayImage(toBufferedImage(img), "found mrz?");

                break;
            }
        }

        return roi;
    }

    protected void draw(final Mat src, final Mat dst) {

        if (dst != src) {
            src.copyTo(dst);
        }

        if (mSceneCorners.height() < 4) {
            // The target has not been found.

            // Draw a thumbnail of the target in the upper-left
            // corner so that the user knows what it is.

            // Compute the thumbnail's larger dimension as half the
            // video frame's smaller dimension.
            int height = mReferenceImage.height();
            int width = mReferenceImage.width();
            final int maxDimension = Math.min(dst.width(),
                    dst.height()) / 2;
            final double aspectRatio = width / (double)height;
            if (height > width) {
                height = maxDimension;
                width = (int)(height * aspectRatio);
            } else {
                width = maxDimension;
                height = (int)(width / aspectRatio);
            }

            // Select the region of interest (ROI) where the thumbnail
            // will be drawn.
            final Mat dstROI = dst.submat(0, height, 0, width);

            // Copy a resized reference image into the ROI.
            Imgproc.resize(mReferenceImage, dstROI, dstROI.size(),
                    0.0, 0.0, Imgproc.INTER_AREA);

            return;
        }

        // Outline the found target in green.
        Imgproc.line(dst, new Point(mSceneCorners.get(0, 0)),
                new Point(mSceneCorners.get(1, 0)), mLineColor, 4);
        Imgproc.line(dst, new Point(mSceneCorners.get(1, 0)),
                new Point(mSceneCorners.get(2, 0)), mLineColor, 4);
        Imgproc.line(dst, new Point(mSceneCorners.get(2, 0)),
                new Point(mSceneCorners.get(3, 0)), mLineColor, 4);
        Imgproc.line(dst, new Point(mSceneCorners.get(3,0)),
                new Point(mSceneCorners.get(0, 0)), mLineColor, 4);
    }

    private double angle(Point pt1, Point pt2, Point pt0)
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2)/Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    }

    /* findSquares: returns sequence of squares detected on the image
     */
    void findSquares(Mat src, List<MatOfPoint> squares)
    {
        Mat src_gray = new Mat();
        Imgproc.cvtColor(src, src_gray, Imgproc.COLOR_BGR2GRAY);

        // Blur helps to decrease the amount of detected edges
        Mat filtered = new Mat();
        //Imgproc.blur(src_gray, filtered, new Size(3, 3));
        filtered = src_gray.clone();
        //displayImage(toBufferedImage(filtered), "blurred");

        // Detect edges
        Mat edges = new Mat();
        int thresh = 50;
        //Imgproc.Canny(filtered, edges, thresh, thresh*2);
        Imgproc.Canny(filtered, edges, thresh, 255);
        //displayImage(toBufferedImage(edges), "edges");


        // Dilate helps to connect nearby line segments
        Mat dilated_edges = new Mat();
        Imgproc.dilate(edges, dilated_edges, new Mat(), new Point(-1, -1), 2, 1, new Scalar(0,255,0)); // default 3x3 kernel
        //displayImage(toBufferedImage(dilated_edges), "dilated edges");

        // Find contours and store them in a list
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(dilated_edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Test contours and assemble squares out of them
        MatOfPoint2f approx = new MatOfPoint2f();

        approx.convertTo(approx, CvType.CV_32F);

        for (int i = 0; i < contours.size(); i++)
        {
            MatOfPoint cntCvt = new MatOfPoint();

            contours.get(i).convertTo(cntCvt, CvType.CV_32F);

            contours.set(i, cntCvt);

            // approximate contour with accuracy proportional to the contour perimeter
            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i)), approx, Imgproc.arcLength(new MatOfPoint2f(contours.get(i)), true)*0.02, true);

            Point [] approx_array = new Point [4];

            approx_array = approx.toArray();
            MatOfPoint approx_matofpoint = new MatOfPoint(approx_array);


            Rect bRectContour = Imgproc.boundingRect(approx_matofpoint);

            //approx_matofpoint = approx_list.
            // Note: absolute value of an area is used because
            // area may be positive or negative - in accordance with the
            // contour orientation
            //ADDED: Check that the height and width of possible candidates
            // are at least 1/2 of the image dimensions
            if (approx_array.length == 4 && Math.abs(Imgproc.contourArea(new MatOfPoint2f(approx))) > 1000 &&
                    Imgproc.isContourConvex(approx_matofpoint))
            {
                double maxCosine = 0;
                for (int j = 2; j < 5; j++)
                {
                    double cosine = Math.abs(angle(approx_array[j%4], approx_array[j-2], approx_array[j-1]));
                    maxCosine = Math.max(maxCosine, cosine);
                }

                if (maxCosine < 0.3)
                    squares.add(new MatOfPoint(approx_array));
            }
        }
    }

    /* findLargestSquare: find the largest square within a set of squares
     */
    private MatOfPoint findLargestSquare(List<MatOfPoint> squares)
    {
        if (squares.size() == 0)
        {
            //std::cout << "findLargestSquare !!! No squares detect, nothing to do." << std::endl;
            return new MatOfPoint();
        }

        int max_width = 0;
        int max_height = 0;
        int max_square_idx = 0;

        for (int i = 0; i < squares.size(); i++)
        {
            // Convert a set of 4 unordered Points into a meaningful cv::Rect structure.
            Rect rectangle = Imgproc.boundingRect(squares.get(i));

            //std::cout << "find_largest_square: #" << i << " rectangle x:" << rectangle.x << " y:" << rectangle.y << " " << rectangle.width << "x" << rectangle.height << endl;

            // Store the index position of the biggest square found
            if ((rectangle.width >= max_width) && (rectangle.height >= max_height))
            {
                max_width = rectangle.width;
                max_height = rectangle.height;
                max_square_idx = i;
            }
        }

        return squares.get(max_square_idx);
    }

    private Mat findRoundedCornersID(Mat src)
    {
        if (src.empty())
        {
            return src;
        }

        if (src.width() > 800)
            while (src.width() > 800)// || img.height() > 600)
                Imgproc.pyrDown(src, src);

        List<MatOfPoint> squares = new ArrayList<MatOfPoint>();

        findSquares(src, squares);

        if (squares.size() == 0)
            return src;

        // Draw all detected squares
        Mat src_squares = src.clone();
        for (int i = 0; i < squares.size(); i++)
        {
            int n = squares.get(i).rows();
            Imgproc.polylines(src_squares,squares, true, new Scalar(0, 255, 0), 2, Core.LINE_AA, 0);
        }

        //displayImage(toBufferedImage(src_squares), "src squares");

        MatOfPoint largest_square = null;
        largest_square = findLargestSquare(squares);

        if (largest_square == null)
            return src;

        List<Point> largest_square_list = new ArrayList<Point>();
        largest_square_list = largest_square.toList();

        // Draw circles at the corners
        for (int i = 0; i < largest_square_list.size(); i++ )
            Imgproc.circle(src, largest_square_list.get(i), 4, new Scalar(255, 255, 255), Core.FILLED);

        //displayImage(toBufferedImage(src), "corners");

        Rect bRect = Imgproc.boundingRect(largest_square);

        src = new Mat(src, bRect);

        //displayImage(toBufferedImage(idInSrc), "cropped to id only");

        return src;
    }
    //static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public String run(Mat img) {

        //detectID(img);
        thisActivity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(thisActivity, "Start search for ID",
                        Toast.LENGTH_SHORT).show();
            }
        });

        Mat detectedID = findRoundedCornersID(img);

        if (detectedID.size() != img.size()) {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "Found possible ID. Starting search for MRZ",
                            Toast.LENGTH_SHORT).show();
                }
            });
        }
        else
        {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "ID not found. Exiting",
                            Toast.LENGTH_SHORT).show();
                }
            });
            return "";
        }


        //Mat detectedID = new Mat(img, new Rect(topLeft, bottomRight));


        Mat mrz = detectMRZ(detectedID);

        //displayImage(toBufferedImage(mrz), "found MRZ?");

        if (mrz.empty()) {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "MRZ Not found. Exiting",
                            Toast.LENGTH_SHORT).show();
                }
            });

            return "";
        }

        thisActivity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(thisContext, "Found MRZ. Running OCR",
                        Toast.LENGTH_SHORT).show();
            }
        });

        TessBaseAPI baseApi = new TessBaseAPI();
        baseApi.setDebug(true);
        baseApi.init(DATA_PATH, "eng");

        Mat result = new Mat();
        Imgproc.cvtColor(mrz, result, Imgproc.COLOR_RGB2BGRA);
        Bitmap bmp = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result, bmp);

        baseApi.setImage(bmp);
        final String recognizedText = baseApi.getUTF8Text();
        baseApi.end();

        //Imgproc.putText(img, recognizedText, new Point(0, img.height()-20), 1, 1.0, new Scalar(0, 255, 0), 2);

        if (recognizedText == "")
        {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "No text recognized",
                            Toast.LENGTH_SHORT).show();
                }
            });
        }
        else
        {
            thisActivity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(thisContext, "OCR successful:\n"+ recognizedText,
                            Toast.LENGTH_LONG).show();
                }
            });
        }
        return recognizedText;
    }

/*    public static void main(String[] args) {
        System.out.println(args[0]);

        try{
            new DetectIDMRZ().run(args[0], args[1]);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }*/
}