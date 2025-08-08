#include "grid_sample_3d_plugin.h"

#include <cstring>
#include <cassert>
#include <iostream>

#include <cuda_fp16.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "grid_sample_3d.h"

using namespace nvinfer1;
using nvinfer1::plugin::GridSample3DPlugin;
using nvinfer1::plugin::GridSample3DPluginCreator;

using half = __half;

// plugin specific constants
namespace
{
    static const AsciiChar *GRID_SAMPLER_PLUGIN_VERSION = "1";
    static const AsciiChar *GRID_SAMPLER_PLUGIN_NAME = "GridSample3D";
    static const AsciiChar *GRID_SAMPLER_PLUGIN_NAMESPACE = "";
} // namespace

PluginFieldCollection GridSample3DPluginCreator::mFC{};
std::vector<PluginField> GridSample3DPluginCreator::mPluginAttributes;

// utility helpers, keep same layout as original serialization
template <typename scalar_t>
void writeToBuffer(char *&buffer, const scalar_t &val)
{
    *reinterpret_cast<scalar_t *>(buffer) = val;
    buffer += sizeof(scalar_t);
}

template <typename scalar_t>
scalar_t readFromBuffer(const char *&buffer)
{
    scalar_t val = *reinterpret_cast<const scalar_t *>(buffer);
    buffer += sizeof(scalar_t);
    return val;
}

// Constructors
GridSample3DPlugin::GridSample3DPlugin(const std::string name,
                                       size_t inputChannel,
                                       size_t inputDepth,
                                       size_t inputHeight,
                                       size_t inputWidth,
                                       size_t gridDepth,
                                       size_t gridHeight,
                                       size_t gridWidth,
                                       bool alignCorners,
                                       GridSample3DInterpolationMode interpolationMode,
                                       GridSample3DPaddingMode paddingMode,
                                       DataType dataType)
    : mLayerName(name),
      mInputChannel(inputChannel),
      mInputDepth(inputDepth),
      mInputHeight(inputHeight),
      mInputWidth(inputWidth),
      mGridDepth(gridDepth),
      mGridHeight(gridHeight),
      mGridWidth(gridWidth),
      mAlignCorners(alignCorners),
      mInterpolationMode(interpolationMode),
      mPaddingMode(paddingMode),
      mDataType(dataType),
      mBatch(0)
{
}

GridSample3DPlugin::GridSample3DPlugin(const std::string name,
                                       bool alignCorners,
                                       GridSample3DInterpolationMode interpolationMode,
                                       GridSample3DPaddingMode paddingMode)
    : mLayerName(name),
      mAlignCorners(alignCorners),
      mInterpolationMode(interpolationMode),
      mPaddingMode(paddingMode),
      mBatch(0),
      mInputChannel(0),
      mInputDepth(0),
      mInputHeight(0),
      mInputWidth(0),
      mGridDepth(0),
      mGridHeight(0),
      mGridWidth(0),
      mDataType(DataType::kFLOAT)
{
}

GridSample3DPlugin::GridSample3DPlugin(const std::string name, const void *buffer, size_t buffer_size)
    : mLayerName(name)
{
    const char *data = reinterpret_cast<const char *>(buffer);
    const char *start = data;
    mInputChannel = readFromBuffer<size_t>(data);
    mInputDepth = readFromBuffer<size_t>(data);
    mInputHeight = readFromBuffer<size_t>(data);
    mInputWidth = readFromBuffer<size_t>(data);
    mGridDepth = readFromBuffer<size_t>(data);
    mGridHeight = readFromBuffer<size_t>(data);
    mGridWidth = readFromBuffer<size_t>(data);
    mAlignCorners = readFromBuffer<bool>(data);
    mInterpolationMode = readFromBuffer<GridSample3DInterpolationMode>(data);
    mPaddingMode = readFromBuffer<GridSample3DPaddingMode>(data);
    mDataType = readFromBuffer<DataType>(data);

    // verify expected size
    assert(static_cast<size_t>(data - start) == (sizeof(size_t) * 7 + sizeof(bool) +
                                                 sizeof(GridSample3DInterpolationMode) + sizeof(GridSample3DPaddingMode) + sizeof(DataType)));
}

GridSample3DPlugin::~GridSample3DPlugin() noexcept {}

// IPluginV3
IPluginCapability *GridSample3DPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild *>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime *>(this);
        }
        // kCORE
        return static_cast<IPluginV3OneCore *>(this);
    }
    catch (...)
    {
        return nullptr;
    }
}

IPluginV3 *GridSample3DPlugin::clone() noexcept
{
    auto plugin = new GridSample3DPlugin(mLayerName,
                                         mInputChannel,
                                         mInputDepth,
                                         mInputHeight,
                                         mInputWidth,
                                         mGridDepth,
                                         mGridHeight,
                                         mGridWidth,
                                         mAlignCorners,
                                         mInterpolationMode,
                                         mPaddingMode,
                                         mDataType);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

// IPluginV3OneBuild methods

int32_t GridSample3DPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t GridSample3DPlugin::getOutputDataTypes(
    DataType *outputTypes, int32_t nbOutputs, const DataType *inputTypes, int32_t nbInputs) const noexcept
{
    // keep same behavior as previous getOutputDataType
    assert(nbOutputs == 1);
    assert(nbInputs >= 1);
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t GridSample3DPlugin::getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const * /*shapeInputs*/, int32_t /*nbShapeInputs*/, DimsExprs *outputs, int32_t nbOutputs, IExprBuilder & /*exprBuilder*/) noexcept
{
    // Mirror old getOutputDimensions logic
    assert(nbInputs >= 2);
    assert(outputs != nullptr);
    assert(inputs[0].nbDims == 5);
    assert(inputs[1].nbDims == 5);

    DimsExprs gridDim = inputs[1];
    DimsExprs output(inputs[0]);
    // layout: input dims: N, C, D, H, W (nbDims=5)
    // grid dims: N, D_grid, H_grid, W_grid, 3  (nbDims=5)
    output.d[2] = gridDim.d[1]; // D_grid
    output.d[3] = gridDim.d[2]; // H_grid
    output.d[4] = gridDim.d[3]; // W_grid
    outputs[0] = output;
    return 0;
}

bool GridSample3DPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const *inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    // same logic as before, adapted to DynamicPluginTensorDesc
    assert(nbInputs == 2 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));

    bool condition = inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    condition &= (inOut[pos].desc.type == nvinfer1::DataType::kFLOAT ||
                  inOut[pos].desc.type == nvinfer1::DataType::kHALF);
    condition &= (inOut[pos].desc.type == inOut[0].desc.type);
    return condition;
}

int32_t GridSample3DPlugin::configurePlugin(DynamicPluginTensorDesc const *in,
                                            int32_t nbInputs,
                                            DynamicPluginTensorDesc const *out,
                                            int32_t nbOutputs) noexcept
{
    // Previously configurePlugin returned void and set dims; now return int32_t
    assert(nbInputs == 2 && nbOutputs == 1);
    // for 3d grid sample, the input should be 5 dims
    assert(in[0].desc.dims.nbDims == 5);
    assert(in[1].desc.dims.nbDims == 5);

    mBatch = in[0].desc.dims.d[0];
    mInputChannel = in[0].desc.dims.d[1];
    mInputDepth = in[0].desc.dims.d[2];
    mInputHeight = in[0].desc.dims.d[3];
    mInputWidth = in[0].desc.dims.d[4];
    mGridDepth = in[1].desc.dims.d[1];
    mGridHeight = in[1].desc.dims.d[2];
    mGridWidth = in[1].desc.dims.d[3];
    mDataType = in[0].desc.type;

    assert(mBatch == in[1].desc.dims.d[0]);
    assert(in[1].desc.dims.d[4] == 3);
    return 0;
}

int32_t GridSample3DPlugin::getWorkspaceSize(PluginTensorDesc const * /*inputs*/,
                                             int32_t /*nbInputs*/,
                                             PluginTensorDesc const * /*outputs*/,
                                             int32_t /*nbOutputs*/) const noexcept
{
    return 0;
}

// IPluginV3OneRuntime methods

int32_t GridSample3DPlugin::onShapeChange(PluginTensorDesc const *in,
                                          int32_t nbInputs,
                                          PluginTensorDesc const *out,
                                          int32_t nbOutputs) noexcept
{
    // Called before enqueue at runtime (mirror configurePlugin semantics for runtime)
    assert(nbInputs == 2 && nbOutputs == 1);
    assert(in[0].dims.nbDims == 5);
    assert(in[1].dims.nbDims == 5);

    mBatch = in[0].dims.d[0];
    mInputChannel = in[0].dims.d[1];
    mInputDepth = in[0].dims.d[2];
    mInputHeight = in[0].dims.d[3];
    mInputWidth = in[0].dims.d[4];
    mGridDepth = in[1].dims.d[1];
    mGridHeight = in[1].dims.d[2];
    mGridWidth = in[1].dims.d[3];
    mDataType = in[0].type;

    assert(mBatch == in[1].dims.d[0]);
    assert(in[1].dims.d[4] == 3);
    return 0;
}

int32_t GridSample3DPlugin::enqueue(PluginTensorDesc const * /*inputDesc*/,
                                    PluginTensorDesc const * /*outputDesc*/,
                                    void const *const *inputs,
                                    void *const *outputs,
                                    void * /*workspace*/,
                                    cudaStream_t stream) noexcept
{
    int status = -1;
    if (mDataType == DataType::kFLOAT)
    {
        status = grid_sample_3d_cuda<float>(
            static_cast<const float *>(inputs[0]),
            static_cast<const float *>(inputs[1]),
            mBatch, mInputChannel, mInputDepth, mInputHeight, mInputWidth,
            mGridDepth, mGridHeight, mGridWidth,
            mAlignCorners,
            mInterpolationMode,
            mPaddingMode,
            static_cast<float *>(outputs[0]),
            stream);
    }
    else if (mDataType == DataType::kHALF)
    {
        status = grid_sample_3d_cuda<half>(
            static_cast<const half *>(inputs[0]),
            static_cast<const half *>(inputs[1]),
            mBatch, mInputChannel, mInputDepth, mInputHeight, mInputWidth,
            mGridDepth, mGridHeight, mGridWidth,
            mAlignCorners,
            mInterpolationMode,
            mPaddingMode,
            static_cast<half *>(outputs[0]),
            stream);
    }

    return status;
}

IPluginV3 *GridSample3DPlugin::attachToContext(IPluginResourceContext * /*context*/) noexcept
{
    // V3 expects this to produce a clone per execution context; plugin does not require per-context resources,
    // so return a clone with same params (minimal change from old attach/detach behavior).
    return clone();
}

// IPluginV3OneCore methods

const AsciiChar *GridSample3DPlugin::getPluginName() const noexcept
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const AsciiChar *GridSample3DPlugin::getPluginVersion() const noexcept
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

const AsciiChar *GridSample3DPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

void GridSample3DPlugin::setPluginNamespace(AsciiChar const *pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace ? pluginNamespace : "";
}

size_t GridSample3DPlugin::getSerializationSize() const noexcept
{
    return sizeof(size_t) * 7 + sizeof(bool) + sizeof(GridSample3DInterpolationMode) + sizeof(GridSample3DPaddingMode) + sizeof(DataType);
}

void GridSample3DPlugin::serialize(void *buffer) const noexcept
{
    char *data = reinterpret_cast<char *>(buffer);
    char *start = data;
    writeToBuffer<size_t>(data, mInputChannel);
    writeToBuffer<size_t>(data, mInputDepth);
    writeToBuffer<size_t>(data, mInputHeight);
    writeToBuffer<size_t>(data, mInputWidth);
    writeToBuffer<size_t>(data, mGridDepth);
    writeToBuffer<size_t>(data, mGridHeight);
    writeToBuffer<size_t>(data, mGridWidth);
    writeToBuffer<bool>(data, mAlignCorners);
    writeToBuffer<GridSample3DInterpolationMode>(data, mInterpolationMode);
    writeToBuffer<GridSample3DPaddingMode>(data, mPaddingMode);
    writeToBuffer<DataType>(data, mDataType);
    assert(static_cast<size_t>(data - start) == getSerializationSize());
}

void GridSample3DPlugin::destroy() noexcept
{
    delete this;
}

nvinfer1::PluginFieldCollection const *GridSample3DPlugin::getFieldsToSerialize() noexcept
{
    // static to ensure it persists after function returns
    static nvinfer1::PluginFieldCollection emptyCollection{};
    emptyCollection.nbFields = 0;
    emptyCollection.fields = nullptr;
    return &emptyCollection;
}

// ---------------- Plugin Creator ----------------

GridSample3DPluginCreator::GridSample3DPluginCreator()
{
    setPluginNamespace(GRID_SAMPLER_PLUGIN_NAMESPACE);
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

GridSample3DPluginCreator::~GridSample3DPluginCreator() noexcept {}

AsciiChar const *GridSample3DPluginCreator::getPluginName() const noexcept
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

AsciiChar const *GridSample3DPluginCreator::getPluginVersion() const noexcept
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

PluginFieldCollection const *GridSample3DPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3 *GridSample3DPluginCreator::createPlugin(AsciiChar const *name, PluginFieldCollection const *fc, TensorRTPhase /*phase*/) noexcept
{
    // Interpret fields same as before
    int interpolationMode = 0;
    int paddingMode = 0;
    int alignCorners = 0;

    if (fc && fc->nbFields > 0)
    {
        const PluginField *fields = fc->fields;
        int nbFields = fc->nbFields;
        for (int i = 0; i < nbFields; ++i)
        {
            const char *field_name = fields[i].name;
            const void *field_data = fields[i].data;
            if (!strcmp(field_name, "interpolation_mode"))
            {
                interpolationMode = *reinterpret_cast<const int *>(field_data);
            }
            else if (!strcmp(field_name, "padding_mode"))
            {
                paddingMode = *reinterpret_cast<const int *>(field_data);
            }
            else if (!strcmp(field_name, "align_corners"))
            {
                alignCorners = *reinterpret_cast<const int *>(field_data);
            }
        }
    }

    std::cout << "paddingMode: " << paddingMode << std::endl;
    std::cout << "interpolationMode: " << interpolationMode << std::endl;

    auto plugin = new GridSample3DPlugin(std::string(name),
                                         static_cast<bool>(alignCorners),
                                         static_cast<GridSample3DInterpolationMode>(interpolationMode),
                                         static_cast<GridSample3DPaddingMode>(paddingMode));
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void GridSample3DPluginCreator::setPluginNamespace(AsciiChar const *libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

AsciiChar const *GridSample3DPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// C-style plugin registration entry points (keep compatibility)
extern "C" TENSORRTAPI IPluginCreatorInterface *const *getCreators(int32_t &nbCreators)
{
    nbCreators = 1;
    static GridSample3DPluginCreator sCreator;
    static IPluginCreatorInterface *const kPLUGIN_CREATOR_LIST[] = {&sCreator};
    return kPLUGIN_CREATOR_LIST;
}

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder *finder)
{
    (void)finder;
}

// Legacy helper (some runtimes still call getPluginCreators)
extern "C" TENSORRTAPI nvinfer1::IPluginCreatorV3One *const *getPluginCreators(int32_t &nbCreators)
{
    nbCreators = 1;
    static GridSample3DPluginCreator sCreator;
    static nvinfer1::IPluginCreatorV3One *const kPLUGIN_CREATOR_LIST[] = {&sCreator};
    return kPLUGIN_CREATOR_LIST;
}

// Register with macro for static registration
REGISTER_TENSORRT_PLUGIN(GridSample3DPluginCreator);
