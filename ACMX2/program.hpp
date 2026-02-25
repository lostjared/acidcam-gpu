#include <gl.hpp>

namespace ac {
    class ShaderProgram : public gl::ShaderProgram {
    public:
        ShaderProgram() = default;
        ShaderProgram(GLuint id) : gl::ShaderProgram(id) {}
        ~ShaderProgram() = default;
        ShaderProgram(const ShaderProgram &) = default;
        ShaderProgram &operator=(const ShaderProgram &) = default;
        ShaderProgram(ShaderProgram &&) noexcept = default;
        ShaderProgram &operator=(ShaderProgram &&) noexcept = default;
        bool loadProgram(const std::string &v, const std::string &f);
        bool loadProgramFromText(const std::string &v, const std::string &f);
    };
}