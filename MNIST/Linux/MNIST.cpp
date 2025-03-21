#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <cstdint>

// ONNX Runtime C++ API (需已安装)
#include <onnxruntime_cxx_api.h>

static constexpr int MNIST_WIDTH = 28;
static constexpr int MNIST_HEIGHT = 28;

//--------------------------------------------------------------------
// Softmax 函数：对向量进行数值稳定的 softmax 运算
//--------------------------------------------------------------------
template <typename T>
static void softmax(T& input) {
    // 1) 找到最大值，用以数值稳定
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;

    // 2) e^(x - rowmax) 并求和
    for (size_t i = 0; i < input.size(); ++i) {
        float val = std::exp(input[i] - rowmax);
        y[i] = val;
        sum += val;
    }
    // 3) 每个元素除以总和
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

//--------------------------------------------------------------------
// 封装 MNIST 模型推理
//--------------------------------------------------------------------
struct MNISTModel {
    MNISTModel(const char* model_path) {
        // 创建 ONNX Runtime 环境和会话
        Ort::SessionOptions session_options;
        session_ = Ort::Session(env_, model_path, session_options);

        // 创建输入、输出张量
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        input_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info, input_image_.data(), input_image_.size(),
            input_shape_.data(), input_shape_.size()
        );

        output_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info, results_.data(), results_.size(),
            output_shape_.data(), output_shape_.size()
        );
    }

    // 运行推理，返回推断结果（数字 0~9）
    int Run() {
        // onnx 模型里对应的输入、输出名称（需与实际模型对应）
        const char* input_names[] = {"Input3"};
        const char* output_names[] = {"Plus214_Output_0"};

        Ort::RunOptions run_options;
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

        // 对获取到的输出做 softmax
        softmax(results_);

        // 找到概率最大的元素下标
        int index = std::distance(results_.begin(),
                                  std::max_element(results_.begin(), results_.end()));
        return index;
    }

    // 计算 28×28 中每个像素的灰度(0 或 1 等)
    // 此处仅简单地把 0/255 作为黑/白，如果需要更精确的笔迹识别，可做卷积或 Gaussian
    void convertImage(const std::vector<uint8_t>& sdl_pixels, int bigWidth, int bigHeight) {
        // 清空输入图像
        std::fill(input_image_.begin(), input_image_.end(), 0.0f);

        // 将 bigWidth × bigHeight 的像素缩放/投影到 28×28
        // 最简单方法：“整块”按比例缩小
        for(int row = 0; row < MNIST_HEIGHT; ++row) {
            for(int col = 0; col < MNIST_WIDTH; ++col) {
                // 这里采用最近邻插值
                int srcY = row * bigHeight / MNIST_HEIGHT;
                int srcX = col * bigWidth  / MNIST_WIDTH;
                // sdl_pixels 里每个像素 4 字节(RGBA)，取灰度
                int idx = (srcY * bigWidth + srcX) * 4;

                // 这里简单判断 R/G/B 都接近 0 就算“黑”
                // 也可取平均
                uint8_t r = sdl_pixels[idx + 0];
                uint8_t g = sdl_pixels[idx + 1];
                uint8_t b = sdl_pixels[idx + 2];

                float val = (r + g + b) / 3.0f; // [0..255]
                // 简单判断：越黑 => val 越小 => 记为 1.0
                // 这里直接做  1.0 - (val / 255)
                input_image_[row*MNIST_WIDTH + col] = (255.0f - val) / 255.0f;
            }
        }
    }

    // 用于存放 28×28 的浮点图像数据
    std::array<float, MNIST_WIDTH*MNIST_HEIGHT> input_image_{};
    // 模型的输出概率（10 个数字的概率分布）
    std::array<float, 10> results_{};

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "mnist-env"};
    Ort::Session session_{nullptr};
    Ort::Value input_tensor_{nullptr};
    Ort::Value output_tensor_{nullptr};

    // 输入形状：N=1, C=1, H=28, W=28
    std::array<int64_t,4> input_shape_{1, 1, MNIST_HEIGHT, MNIST_WIDTH};
    // 输出形状：1 x 10
    std::array<int64_t,2> output_shape_{1, 10};
};

//--------------------------------------------------------------------
// 主函数：SDL2 窗口，允许用户在画板书写，然后调用 ONNX Runtime 推理
//--------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // 模型路径可自定义，如 "./mnist.onnx"
    const char* model_path = "mnist.onnx";
    std::unique_ptr<MNISTModel> mnistModel;

    try {
        mnistModel = std::make_unique<MNISTModel>(model_path);
    } catch(const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }

    // SDL 初始化
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL cannot init: " << SDL_GetError() << std::endl;
        return -1;
    }

    // 选个大一点的画板，比如 112 = 28*4 倍
    const int width = MNIST_WIDTH * 10;
    const int height = MNIST_HEIGHT * 8;

    SDL_Window* window = SDL_CreateWindow(
        "MNIST Drawing - WSL2 Demo",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_SHOWN
    );
    if (!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

   // 创建渲染器  
   SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);  
   if(!renderer) {  
       std::cerr << "Failed to create renderer: " << SDL_GetError() << std::endl;  
       SDL_DestroyWindow(window);  
       SDL_Quit();  
       return -1;  
   }  
       // MODIFIED: 初始化渲染参数  
       SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);  
       SDL_RenderSetScale(renderer, 1.5f, 1.5f);  // 增加线条粗细  

    // 创建用于绘制的纹理(ARGB8888)
    SDL_Texture* texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_RGBA8888,
                                             SDL_TEXTUREACCESS_TARGET,
                                             width, height);

    // 清空为白色背景
    SDL_SetRenderTarget(renderer, texture);
    SDL_SetRenderDrawColor(renderer, 255,255,255,255);
    SDL_RenderClear(renderer);
    SDL_SetRenderTarget(renderer, NULL);

    bool quit = false;
    bool drawing = false;
    SDL_Point lastPos{0,0};

    std::cout << "Left-click to draw, right-click to clear. Press ESC or close window to quit.\n";

    while(!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch(e.type) {
                case SDL_QUIT:
                    quit = true;
                    break;
                case SDL_KEYDOWN:
                    if (e.key.keysym.sym == SDLK_ESCAPE) {
                        quit = true;
                    }
                    break;
                case SDL_MOUSEBUTTONDOWN:
                if (e.button.button == SDL_BUTTON_LEFT) {  
                    // MODIFIED: 添加坐标校验  
                    lastPos.x = std::clamp(e.button.x, 0, width-1);  
                    lastPos.y = std::clamp(e.button.y, 0, height-1);  
                    drawing = true;  
                }   else if (e.button.button == SDL_BUTTON_RIGHT) {
                        // 右键清空画布
                        SDL_SetRenderTarget(renderer, texture);
                        SDL_SetRenderDrawColor(renderer, 255,255,255,255);
                        SDL_RenderClear(renderer);
                        SDL_SetRenderTarget(renderer, NULL);
                    }
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (e.button.button == SDL_BUTTON_LEFT) {
                        drawing = false;
                        // 松开鼠标后，可以进行推理
                        // 1) 读取当前纹理数据
                        std::vector<uint8_t> pixels(width*height*4); // RGBA
                        SDL_SetRenderTarget(renderer, texture);
                        SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_RGBA8888, pixels.data(), width*4);
                        SDL_SetRenderTarget(renderer, NULL);

                        // 2) 转换到 mnist 输入
                        mnistModel->convertImage(pixels, width, height);

                        // 3) 模型推理
                        int predicted = mnistModel->Run();

                        // 4) 打印结果
                        std::cout << "Predicted digit index: " << predicted << "\nProbabilities:\n";
                        for (int i = 0; i < 10; i++) {
                            std::cout << "  " << i << ": " << mnistModel->results_[i] << std::endl;
                        }
                    }
                    break;
                case SDL_MOUSEMOTION:
                    if (drawing) {
                        // MODIFIED: 更新绘图逻辑  
                        int x = std::clamp(e.motion.x, 0, width-1);  
                        int y = std::clamp(e.motion.y, 0, height-1);  

                        SDL_SetRenderTarget(renderer, texture);  
                        
                        // 绘制粗线条  
                        SDL_RenderDrawLine(renderer, lastPos.x, lastPos.y, x, y);  
                        for(int i = -2; i <= 2; ++i) {  // 5像素宽  
                            SDL_RenderDrawLine(renderer,   
                                lastPos.x + i, lastPos.y,   
                                x + i, y);  
                        }  
                        
                        SDL_SetRenderTarget(renderer, NULL);  
                        lastPos = {x, y}; 
                    }
                    break;
            }
        }

        // 每帧渲染
        SDL_SetRenderDrawColor(renderer, 200,200,200,255);
        SDL_RenderClear(renderer);

        // 将画板纹理贴到窗口上
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        SDL_Delay(10); // 降低 CPU 占用
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
