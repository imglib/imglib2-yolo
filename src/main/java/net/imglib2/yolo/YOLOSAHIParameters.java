package net.imglib2.yolo;

import java.util.HashMap;
import java.util.Map;

import net.imglib2.appose.ShmImg;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

public class YOLOSAHIParameters
{

    // ── Model ─────────────────────────────────────────────────────────────────
    public final YOLOBuiltinModels builtinModel;

    public final String customModel;

    // ── Core inference ────────────────────────────────────────────────────────
    public final double conf;

    // ── SAHI ──────────────────────────────────────────────────────────────────
    public final boolean useSahi;

    public final int sliceHeight;

    public final int sliceWidth;

    public final double overlapHeightRatio;

    public final double overlapWidthRatio;

    public final boolean performStandardPred;

    public final String postprocessType;

    public final double postprocessMatchThreshold;

    // ── Advanced ──────────────────────────────────────────────────────────────
    public final int minArea;

    // ── GPU ───────────────────────────────────────────────────────────────────
    public final boolean useGpu;

    private YOLOSAHIParameters( final Builder b )
    {
        this.builtinModel                = b.builtinModel;
        this.customModel                 = b.customModel;
        this.conf                        = b.conf;
        this.useSahi                     = b.useSahi;
        this.sliceHeight                 = b.sliceHeight;
        this.sliceWidth                  = b.sliceWidth;
        this.overlapHeightRatio          = b.overlapHeightRatio;
        this.overlapWidthRatio           = b.overlapWidthRatio;
        this.performStandardPred         = b.performStandardPred;
        this.postprocessType             = b.postprocessType;
        this.postprocessMatchThreshold   = b.postprocessMatchThreshold;
        this.minArea                     = b.minArea;
        this.useGpu                      = b.useGpu;
    }

    /**
     * Builds the map passed to the Appose Python task.
     * Detections are returned via task.export() on the Python side,
     * so no output shared-memory image is needed.
     */
	public < T extends RealType< T > & NativeType< T > > Map< String, Object > toApposeMap( final ShmImg< T > input )
    {
        final Map< String, Object > inputs = new HashMap<>();

        // ── Input image ───────────────────────────────────────────────────────
        inputs.put( "input", input.ndArray() );

        // ── Model ─────────────────────────────────────────────────────────────
        final boolean useCustom = customModel != null && !customModel.isBlank();
        inputs.put( "model_file", useCustom ? customModel : builtinModel.modelFile() );

        // ── Core inference ────────────────────────────────────────────────────
        inputs.put( "conf",  conf );

        // ── SAHI ──────────────────────────────────────────────────────────────
        inputs.put( "use_sahi",                    useSahi );
        inputs.put( "slice_height",                sliceHeight );
        inputs.put( "slice_width",                 sliceWidth );
        inputs.put( "overlap_height_ratio",        overlapHeightRatio );
        inputs.put( "overlap_width_ratio",         overlapWidthRatio );
        inputs.put( "perform_standard_pred",       performStandardPred );
        inputs.put( "postprocess_type",            postprocessType );
        inputs.put( "postprocess_match_threshold", postprocessMatchThreshold );

        // ── Advanced ──────────────────────────────────────────────────────────
        inputs.put( "min_area", minArea );

        // ── GPU ───────────────────────────────────────────────────────────────
        inputs.put( "use_gpu", useGpu );

        return inputs;
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    public static Builder builder() { return new Builder(); }

    public static class Builder
    {
        private YOLOBuiltinModels builtinModel           = YOLOBuiltinModels.YOLO26L;
        private String customModel                       = null;
        private double conf                              = 0.25;
        private boolean useSahi                          = true;
        private int    sliceHeight                       = 640;
        private int    sliceWidth                        = 640;
        private double overlapHeightRatio                = 0.2;
        private double overlapWidthRatio                 = 0.2;
        private boolean performStandardPred              = true;
        private String postprocessType                   = "NMS";
        private double postprocessMatchThreshold         = 0.5;
        private int    minArea                           = 0;
        private boolean useGpu                           = true;

        public Builder builtinModel( final YOLOBuiltinModels m )    { builtinModel = m;  return this; }
        public Builder customModel( final String p )                { customModel = p;   return this; }
        public Builder conf( final double v )                       { conf = v;          return this; }
        public Builder useSahi( final boolean v )                   { useSahi = v;       return this; }
        public Builder sliceSize( final int h, final int w )              { sliceHeight=h; sliceWidth=w; return this; }
        public Builder overlapRatios( final double h, final double w )    { overlapHeightRatio=h; overlapWidthRatio=w; return this; }
        public Builder performStandardPred( final boolean v )       { performStandardPred = v; return this; }
        public Builder postprocessType( final String v )            { postprocessType = v; return this; }
        public Builder postprocessMatchThreshold( final double v )  { postprocessMatchThreshold = v; return this; }
        public Builder minArea( final int v )                       { minArea = v;       return this; }
        public Builder useGpu( final boolean v )                    { useGpu = v;        return this; }

        public YOLOSAHIParameters build()                     { return new YOLOSAHIParameters( this ); }
    }
}
