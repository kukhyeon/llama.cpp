#include "arg.h"
#include "common.h"
#include "download.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    // negated arg
    argv = {"binary_name", "--no-mmap"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);

    common_params hardware_stats_params;
    assert(hardware_stats_params.hardware_stats == true);
    assert(hardware_stats_params.hardware_stats_core == -1);

    argv = {"binary_name", "-m", "model.gguf", "--hardware-stats", "off", "--hardware-stats-core", "4"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), hardware_stats_params, LLAMA_EXAMPLE_COMPLETION));
    assert(hardware_stats_params.hardware_stats == false);
    assert(hardware_stats_params.hardware_stats_core == 4);

    argv = {"binary_name", "-m", "model.gguf", "--hardware-stats", "invalid"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), hardware_stats_params, LLAMA_EXAMPLE_COMPLETION));

    argv = {"binary_name", "-m", "model.gguf", "--hardware-stats-core", "-2"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), hardware_stats_params, LLAMA_EXAMPLE_COMPLETION));

    common_params query_period_default_params;
    assert(query_period_default_params.query_interval == 0);

    common_params query_period_positive_params;
    argv = {"binary_name", "-m", "model.gguf", "--query-period-ms", "60000"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_period_positive_params, LLAMA_EXAMPLE_COMPLETION));
    assert(query_period_positive_params.query_interval == 60000);

    common_params query_period_zero_params;
    argv = {"binary_name", "-m", "model.gguf", "--query-period-ms", "0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_period_zero_params, LLAMA_EXAMPLE_COMPLETION));
    assert(query_period_zero_params.query_interval == 0);

    common_params query_period_negative_params;
    argv = {"binary_name", "-m", "model.gguf", "--query-period-ms", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_period_negative_params, LLAMA_EXAMPLE_COMPLETION));

    common_params query_period_non_completion_params;
    argv = {"binary_name", "--query-period-ms", "60000"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_period_non_completion_params, LLAMA_EXAMPLE_COMMON));

    common_params query_prewake_default_params;
    assert(query_prewake_default_params.query_prewake_us == 0);

    common_params query_prewake_positive_params;
    argv = {"binary_name", "-m", "model.gguf", "--query-prewake-us", "500"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_prewake_positive_params, LLAMA_EXAMPLE_COMPLETION));
    assert(query_prewake_positive_params.query_prewake_us == 500);

    common_params query_prewake_zero_params;
    argv = {"binary_name", "-m", "model.gguf", "--query-prewake-us", "0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_prewake_zero_params, LLAMA_EXAMPLE_COMPLETION));
    assert(query_prewake_zero_params.query_prewake_us == 0);

    common_params query_prewake_negative_params;
    argv = {"binary_name", "-m", "model.gguf", "--query-prewake-us", "-1"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_prewake_negative_params, LLAMA_EXAMPLE_COMPLETION));

    common_params query_prewake_non_completion_params;
    argv = {"binary_name", "--query-prewake-us", "500"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), query_prewake_non_completion_params, LLAMA_EXAMPLE_COMMON));

    common_params attn_qkv_default_params;
    assert(attn_qkv_default_params.attn_qkv_parallel == false);

    common_params attn_qkv_on_params;
    argv = {"binary_name", "-m", "model.gguf", "--attn-qkv-parallel", "on"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_on_params, LLAMA_EXAMPLE_COMPLETION));
    assert(attn_qkv_on_params.attn_qkv_parallel == true);

    common_params attn_qkv_off_params;
    argv = {"binary_name", "-m", "model.gguf", "--attn-qkv-parallel", "off"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_off_params, LLAMA_EXAMPLE_COMPLETION));
    assert(attn_qkv_off_params.attn_qkv_parallel == false);

    common_params attn_qkv_invalid_params;
    argv = {"binary_name", "-m", "model.gguf", "--attn-qkv-parallel", "invalid"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_invalid_params, LLAMA_EXAMPLE_COMPLETION));

    common_params attn_qkv_shards_default_params;
    assert(attn_qkv_shards_default_params.attn_qkv_shards == false);
    const llama_context_params attn_qkv_api_default_params = llama_context_default_params();
    assert(attn_qkv_api_default_params.attn_qkv_shards == false);

    common_params attn_qkv_shards_on_params;
    argv = {"binary_name", "-m", "model.gguf", "--attn-qkv-shards", "on"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_shards_on_params, LLAMA_EXAMPLE_COMPLETION));
    assert(attn_qkv_shards_on_params.attn_qkv_shards == true);

    common_params attn_qkv_shards_off_params;
    argv = {"binary_name", "-m", "model.gguf", "--attn-qkv-shards", "off"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_shards_off_params, LLAMA_EXAMPLE_COMPLETION));
    assert(attn_qkv_shards_off_params.attn_qkv_shards == false);

    common_params attn_qkv_shards_invalid_params;
    argv = {"binary_name", "-m", "model.gguf", "--attn-qkv-shards", "invalid"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_shards_invalid_params, LLAMA_EXAMPLE_COMPLETION));

    common_params attn_qkv_modes_conflict_params;
    argv = {
        "binary_name", "-m", "model.gguf",
        "--attn-qkv-parallel", "on",
        "--attn-qkv-shards", "on",
    };
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_modes_conflict_params, LLAMA_EXAMPLE_COMPLETION));

    common_params attn_qkv_modes_non_conflict_params;
    argv = {
        "binary_name", "-m", "model.gguf",
        "--attn-qkv-parallel", "off",
        "--attn-qkv-shards", "on",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_modes_non_conflict_params, LLAMA_EXAMPLE_COMPLETION));
    assert(attn_qkv_modes_non_conflict_params.attn_qkv_parallel == false);
    assert(attn_qkv_modes_non_conflict_params.attn_qkv_shards == true);

    common_params max_query_unlimited_params;
    argv = {"binary_name", "-m", "model.gguf", "--max-query-number", "-1"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), max_query_unlimited_params, LLAMA_EXAMPLE_COMPLETION));
    assert(max_query_unlimited_params.max_query_number == -1);

    common_params max_query_invalid_params;
    argv = {"binary_name", "-m", "model.gguf", "--max-query-number", "-2"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), max_query_invalid_params, LLAMA_EXAMPLE_COMPLETION));

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_ATTN_QKV_SHARDS", "on", true);
    common_params attn_qkv_shards_env_params;
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_shards_env_params, LLAMA_EXAMPLE_COMPLETION));
    assert(attn_qkv_shards_env_params.attn_qkv_shards == true);
    unsetenv("LLAMA_ARG_ATTN_QKV_SHARDS");

    setenv("LLAMA_ARG_ATTN_QKV_PARALLEL", "on", true);
    setenv("LLAMA_ARG_ATTN_QKV_SHARDS", "on", true);
    common_params attn_qkv_modes_env_conflict_params;
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), attn_qkv_modes_env_conflict_params, LLAMA_EXAMPLE_COMPLETION));
    unsetenv("LLAMA_ARG_ATTN_QKV_PARALLEL");
    unsetenv("LLAMA_ARG_ATTN_QKV_SHARDS");

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_MMAP", "0", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.use_mmap == false);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}
