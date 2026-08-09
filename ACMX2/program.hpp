#include <gl.hpp>

namespace ac {
    /** Restore a compute-only program binary keyed by source and GL driver. */
    bool loadComputeProgramBinaryFromCache(const std::string &source,
                                           GLuint &program);

    /** Save a linked compute-only program to its stage-specific cache key. */
    bool saveComputeProgramBinaryToCache(const std::string &source,
                                         GLuint program);

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
} // namespace ac
