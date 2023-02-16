/*global jQuery */


// $("#subject_label").change(function () {
//     var subjectArray = ['corazao', $("#subject_label").val()];
//     $("#subject_uri").text(subjectArray.join(' '));
// });

$(document).ready(function () {
    $('#predicate_label').select2({  // init Select2 on form's name field
        placeholder: "Please select a predicate",
        allowClear: true,
        tags: true,
        "width": "style"
    });
});