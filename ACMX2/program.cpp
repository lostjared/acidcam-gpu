#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "program.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
    static uint64_t fnv1a64_bytes(const void *data, size_t n) {
        const uint8_t *p = (const uint8_t *)data;
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    static uint64_t fnv1a64_str(const std::string &s) {
        return fnv1a64_bytes(s.data(), s.size());
    }

    static uint64_t fnv1a64_file(const std::string &filepath) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f)
            return 0;

        uint64_t h = 1469598103934665603ull;
        char buf[16384];

        while (f.read(buf, sizeof(buf)) || f.gcount()) {
            std::streamsize n = f.gcount();
            for (std::streamsize i = 0; i < n; ++i) {
                h ^= (uint8_t)buf[i];
                h *= 1099511628211ull;
            }
        }
        return h;
    }

    static uint64_t mix64(uint64_t a, uint64_t b) {
        uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
        x ^= (x >> 33);
        x *= 0xff51afd7ed558ccdull;
        x ^= (x >> 33);
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= (x >> 33);
        return x;
    }

    static std::string glStr(GLenum e) {
        const GLubyte *p = glGetString(e);
        if (!p)
            return {};
        return std::string((const char *)p);
    }

    struct CacheHeader {
        uint32_t magic;
        uint16_t version;
        uint16_t reserved;
        uint64_t key;
        uint32_t binaryFormat;
        uint32_t binaryLength;
    };

    static constexpr uint32_t kMagic = 0x53434143u;
    static constexpr uint16_t kVersion = 1;

    static bool readWholeFile(const std::string &path, std::vector<uint8_t> &out) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return false;
        f.seekg(0, std::ios::end);
        std::streamoff sz = f.tellg();
        if (sz <= 0)
            return false;
        f.seekg(0, std::ios::beg);
        out.resize((size_t)sz);
        f.read((char *)out.data(), (std::streamsize)out.size());
        return f.good();
    }

    static bool writeFileAtomic(const std::string &path, const void *data, size_t size) {
        std::filesystem::path p(path);
        std::filesystem::create_directories(p.parent_path());

        std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
                return false;
            out.write((const char *)data, (std::streamsize)size);
            if (!out.good())
                return false;
        }

        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(tmp, path, ec);
            if (ec) {
                std::filesystem::remove(tmp, ec);
                return false;
            }
        }
        return true;
    }

    static std::string cacheDirDefault() {
        return "shader_cache";
    }

    static std::string cacheFilePath(uint64_t key) {
        char name[64];
        std::snprintf(name, sizeof(name), "%016llx.bin", (unsigned long long)key);
        std::filesystem::path p(cacheDirDefault());
        p /= name;
        return p.string();
    }

    static uint64_t computeProgramKeyFromFiles(const std::string &vsPath, const std::string &fsPath) {
        uint64_t hv = fnv1a64_file(vsPath);
        uint64_t hf = fnv1a64_file(fsPath);

        uint64_t h = mix64(hv, hf);

        std::string vendor = glStr(GL_VENDOR);
        std::string renderer = glStr(GL_RENDERER);
        std::string version = glStr(GL_VERSION);

        h = mix64(h, fnv1a64_str(vendor));
        h = mix64(h, fnv1a64_str(renderer));
        h = mix64(h, fnv1a64_str(version));
        h = mix64(h, fnv1a64_str("ShaderProgramCache_v1"));

        return h;
    }

    static uint64_t computeProgramKeyFromText(const std::string &vs, const std::string &fs) {
        uint64_t hv = fnv1a64_str(vs);
        uint64_t hf = fnv1a64_str(fs);

        uint64_t h = mix64(hv, hf);

        std::string vendor = glStr(GL_VENDOR);
        std::string renderer = glStr(GL_RENDERER);
        std::string version = glStr(GL_VERSION);

        h = mix64(h, fnv1a64_str(vendor));
        h = mix64(h, fnv1a64_str(renderer));
        h = mix64(h, fnv1a64_str(version));
        h = mix64(h, fnv1a64_str("ShaderProgramCache_v1"));

        return h;
    }

    static bool tryLoadProgramBinary(uint64_t key, GLuint &outProg) {
        std::vector<uint8_t> data;
        std::string path = cacheFilePath(key);
        if (!readWholeFile(path, data))
            return false;
        if (data.size() < sizeof(CacheHeader))
            return false;

        CacheHeader hdr{};
        std::memcpy(&hdr, data.data(), sizeof(CacheHeader));
        if (hdr.magic != kMagic)
            return false;
        if (hdr.version != kVersion)
            return false;
        if (hdr.key != key)
            return false;
        if (hdr.binaryLength == 0)
            return false;

        size_t need = sizeof(CacheHeader) + (size_t)hdr.binaryLength;
        if (data.size() != need)
            return false;

        GLuint prog = glCreateProgram();
        glProgramBinary(prog, (GLenum)hdr.binaryFormat, data.data() + sizeof(CacheHeader), (GLsizei)hdr.binaryLength);

        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            glDeleteProgram(prog);
            return false;
        }

        outProg = prog;
        return true;
    }

    static bool saveProgramBinary(uint64_t key, GLuint prog) {
        GLint binLen = 0;
        glGetProgramiv(prog, GL_PROGRAM_BINARY_LENGTH, &binLen);
        if (binLen <= 0)
            return false;

        std::vector<uint8_t> blob;
        blob.resize(sizeof(CacheHeader) + (size_t)binLen);

        CacheHeader hdr{};
        hdr.magic = kMagic;
        hdr.version = kVersion;
        hdr.key = key;

        GLenum fmt = 0;
        GLsizei got = 0;
        glGetProgramBinary(prog, (GLsizei)binLen, &got, &fmt, blob.data() + sizeof(CacheHeader));
        if (got <= 0)
            return false;

        hdr.binaryFormat = (uint32_t)fmt;
        hdr.binaryLength = (uint32_t)got;

        std::memcpy(blob.data(), &hdr, sizeof(CacheHeader));

        return writeFileAtomic(cacheFilePath(key), blob.data(), blob.size());
    }

} // namespace

namespace ac {

    bool ShaderProgram::loadProgram(const std::string &v, const std::string &f) {
        uint64_t key = computeProgramKeyFromFiles(v, f);
        GLuint cached = 0;
        if (tryLoadProgramBinary(key, cached)) {
            static_cast<gl::ShaderProgram &>(*this) = gl::ShaderProgram(cached);
            return true;
        }
        if (!gl::ShaderProgram::loadProgram(v, f))
            return false;
        saveProgramBinary(key, id());
        return true;
    }

    bool ShaderProgram::loadProgramFromText(const std::string &v, const std::string &f) {
        uint64_t key = computeProgramKeyFromText(v, f);
        GLuint cached = 0;
        if (tryLoadProgramBinary(key, cached)) {
            static_cast<gl::ShaderProgram &>(*this) = gl::ShaderProgram(cached);
            return true;
        }
        if (!gl::ShaderProgram::loadProgramFromText(v, f))
            return false;
        saveProgramBinary(key, id());
        return true;
    }
} // namespace ac
