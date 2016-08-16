package com.example.chris.secondsight.filters;

import org.opencv.core.Mat;

/**
 * Created by chris on 8/8/16.
 */
public interface Filter {
    public abstract void apply(final Mat src, final Mat dst);
}
