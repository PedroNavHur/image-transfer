module.exports = {
  extends: ["stylelint-config-standard", "stylelint-config-tailwindcss"],
  rules: {
    "at-rule-no-unknown": [
      true,
      { ignoreAtRules: ["tailwind", "plugin", "theme"] },
    ],
    "hue-degree-notation": null,
  },
};
