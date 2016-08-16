package com.example.chris.secondsight.filters.mixer;

import com.example.chris.secondsight.filters.Filter;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.ArrayList;

/**
 * Created by chris on 8/8/16.
 */
public class RecolorRCFilter implements Filter {

    private final ArrayList<Mat> mChannels = new ArrayList<Mat>(4);

    @Override
    public void apply(final Mat src, final Mat dst) {
        Core.split(src, mChannels);
        final Mat g = mChannels.get(1);
        final Mat b = mChannels.get(2);
// dst.g = 0.5 * src.g + 0.5 * src.b
        Core.addWeighted(g, 0.5, b, 0.5, 0.0, g);
// dst.b = dst.g
        mChannels.set(2, g);
        Core.merge(mChannels, dst);
    }
}
