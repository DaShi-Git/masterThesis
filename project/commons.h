#pragma once

/*
 * COMMON MACROS
 */

// Common namespace of the renderer library
#define BEGIN_RENDERER_NAMESPACE namespace renderer {
#define END_RENDERER_NAMESPACE }
#define RENDERER_NAMESPACE ::renderer

// DLL Import and Export
#define _LIB_BUILD_SHARED_LIBS
#ifdef _WIN32
#if defined(_LIB_BUILD_SHARED_LIBS)
#define _LIB_EXPORT __declspec(dllexport)
#define _LIB_IMPORT __declspec(dllimport)
#else
#define _LIB_EXPORT
#define _LIB_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define _LIB_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define _LIB_EXPORT
#endif // defined(__GNUC__)
#define _LIB_IMPORT _LIB_EXPORT
#endif // _WIN32

#ifdef RENDERER_BUILD_SHARED
#ifdef BUILD_MAIN_LIB
#define MY_API _LIB_EXPORT
#else
#define MY_API _LIB_IMPORT
#endif
#else 
#define MY_API
#endif

BEGIN_RENDERER_NAMESPACE
class NonAssignable {
    //https://stackoverflow.com/a/22495199
public:
    NonAssignable(NonAssignable const&) = delete;
    NonAssignable& operator=(NonAssignable const&) = delete;
    NonAssignable() {}
};
END_RENDERER_NAMESPACE

#define RENDERER_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                      \
    using scalar_t = type;                               \
    return __VA_ARGS__();                                \
  }
#define RENDERER_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                           \
    const c10::ScalarType the_type = TYPE;                                        \
    switch (the_type) {                                                           \
      RENDERER_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)     \
      RENDERER_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)       \
      default:                                                                    \
        AT_ERROR(#NAME, " not implemented for '", toString(the_type), "'");       \
    }                                                                             \
  }()

#ifdef _MSC_VER
#define release_assert(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )
#else
#define release_assert assert
#endif
