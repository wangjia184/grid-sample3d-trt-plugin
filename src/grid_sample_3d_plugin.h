#pragma once

#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <grid_sample_3d.h> // your CUDA kernel declarations

#ifndef GRID_SAMPLE_3D_PLUGIN
#define GRID_SAMPLE_3D_PLUGIN

using namespace nvinfer1;

namespace nvinfer1
{
    namespace plugin
    {

        class GridSample3DPlugin : public IPluginV3,
                                   public IPluginV3OneCore,
                                   public IPluginV3OneBuild,
                                   public IPluginV3OneRuntime
        {
        public:
            GridSample3DPlugin(const std::string name,
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
                               nvinfer1::DataType dataType);

            GridSample3DPlugin(const std::string name,
                               bool alignCorners,
                               GridSample3DInterpolationMode interpolationMode,
                               GridSample3DPaddingMode paddingMode);

            GridSample3DPlugin(const std::string name,
                               const void *buffer,
                               size_t buffer_size);

            GridSample3DPlugin() = delete;
            ~GridSample3DPlugin() noexcept override;

            // IPluginV3
            IPluginCapability *getCapabilityInterface(PluginCapabilityType type) noexcept override;
            IPluginV3 *clone() noexcept override;

            // IPluginV3OneBuild methods (build-time capabilities)
            int32_t getNbOutputs() const noexcept override;
            int32_t getOutputDataTypes(
                DataType *outputTypes, int32_t nbOutputs, const DataType *inputTypes, int32_t nbInputs) const noexcept override;
            int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs, int32_t nbShapeInputs,
                                    DimsExprs *outputs, int32_t nbOutputs, IExprBuilder &exprBuilder) noexcept override;

            bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
            int32_t configurePlugin(DynamicPluginTensorDesc const *in,
                                    int32_t nbInputs,
                                    DynamicPluginTensorDesc const *out,
                                    int32_t nbOutputs) noexcept override;
            int32_t getWorkspaceSize(PluginTensorDesc const *inputs,
                                     int32_t nbInputs,
                                     PluginTensorDesc const *outputs,
                                     int32_t nbOutputs) const noexcept;

            // IPluginV3OneRuntime methods (runtime capabilities)
            int32_t onShapeChange(PluginTensorDesc const *in,
                                  int32_t nbInputs,
                                  PluginTensorDesc const *out,
                                  int32_t nbOutputs) noexcept override;
            int32_t enqueue(PluginTensorDesc const *inputDesc,
                            PluginTensorDesc const *outputDesc,
                            void const *const *inputs,
                            void *const *outputs,
                            void *workspace,
                            cudaStream_t stream) noexcept override;
            IPluginV3 *attachToContext(IPluginResourceContext *context) noexcept override;

            // IPluginV3OneCore methods (core capabilities)
            const AsciiChar *getPluginName() const noexcept override;
            const AsciiChar *getPluginVersion() const noexcept override;
            const AsciiChar *getPluginNamespace() const noexcept override;
            void setPluginNamespace(AsciiChar const *pluginNamespace) noexcept;
            size_t getSerializationSize() const noexcept;
            void serialize(void *buffer) const noexcept;
            void destroy() noexcept;

            nvinfer1::PluginFieldCollection const *getFieldsToSerialize() noexcept override;

        private:
            // internal parameters
            const std::string mLayerName;
            size_t mBatch;
            size_t mInputChannel, mInputDepth, mInputWidth, mInputHeight;
            size_t mGridDepth, mGridWidth, mGridHeight;
            bool mAlignCorners;
            std::string mNameSpace;
            GridSample3DInterpolationMode mInterpolationMode;
            GridSample3DPaddingMode mPaddingMode;
            nvinfer1::DataType mDataType;
        };

        class GridSample3DPluginCreator : public IPluginCreatorV3One
        {
        public:
            GridSample3DPluginCreator();
            ~GridSample3DPluginCreator() noexcept override;

            // IPluginCreatorV3One methods
            IPluginV3 *createPlugin(AsciiChar const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept override;
            PluginFieldCollection const *getFieldNames() noexcept override;
            AsciiChar const *getPluginName() const noexcept override;
            AsciiChar const *getPluginVersion() const noexcept override;
            void setPluginNamespace(AsciiChar const *libNamespace) noexcept;
            AsciiChar const *getPluginNamespace() const noexcept override;

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
        };

    } // namespace plugin
} // namespace nvinfer1

#endif // GRID_SAMPLE_3D_PLUGIN
