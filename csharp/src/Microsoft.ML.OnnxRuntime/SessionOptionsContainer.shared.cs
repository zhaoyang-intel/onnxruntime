// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime
{
    public static class SessionOptionsContainer
    {
        static Lazy<Action<SessionOptions>> _defaultHandler;

        static readonly Dictionary<string, Lazy<Action<SessionOptions>>> _configurationHandlers =
            new Dictionary<string, Lazy<Action<SessionOptions>>>();

        static Lazy<Action<SessionOptions>> DefaultHandler =>
            _defaultHandler != null
                ? _defaultHandler
                : (_defaultHandler = new Lazy<Action<SessionOptions>>(() => (options) => { /* use as is */ }));

        public static void Register(Action<SessionOptions> defaultHandler) => _defaultHandler =
            new Lazy<Action<SessionOptions>>(() => defaultHandler);

        public static void Register(string configuration, Action<SessionOptions> handler) =>
            _configurationHandlers[configuration] = new Lazy<Action<SessionOptions>>(() => handler);

        public static SessionOptions Create(string configuration = null, bool useDefaultAsFallback = true) =>
            new SessionOptions().ApplyConfiguration(configuration, useDefaultAsFallback);

        public static void Reset()
        {
            _defaultHandler = null;
            _configurationHandlers.Clear();
        }

        public static SessionOptions ApplyConfiguration(this SessionOptions options, string configuration = null,
                                                        bool useDefaultAsFallback = true)
        {
            var handler = Resolve(configuration, useDefaultAsFallback);
            handler(options);

            return options;
        }

        static Action<SessionOptions> Resolve(string configuration = null, bool useDefaultAsFallback = true)
        {
            // Non-scoped services
            {
                if (string.IsNullOrWhiteSpace(configuration))
                    return DefaultHandler.Value;

                if (_configurationHandlers.TryGetValue(configuration, out var handler))
                    return handler.Value;

                if (useDefaultAsFallback)
                    return DefaultHandler.Value;

                throw new KeyNotFoundException($"Configuration not found for '{configuration}'");
            }
        }
    }
}
