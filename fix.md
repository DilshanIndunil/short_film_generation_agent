ඔව්, Google Colab Pro වල වුණත් Continuous Session Limit (උපරිම එක දිගට ධාවනය වන කාලය) සාමාන්‍යයෙන් පැය 12–24 අතර සීමා වේ.

එම කාලය අවසන් වූ පසු (හෝ Compute Units ඉවර වී Session එක Disconnect වූ පසු) Colab එකේ ඇති Temporary Storage එක (RAM සහ Temporary Disk Space) සම්පූර්ණයෙන්ම Reset (Wipe) වේ.

ඒ වෙලාවට නැවත මුල සිට Download කරන්න ඕනද?
නැත! Google Drive එක Mount කර තිබේ නම් නැවත Download කිරීමට අවශ්‍ය නැත.

ඔබ Wan 2.2 Model file එක (14GB file එක) Google Drive එක තුළ Save කර තිබේ නම්:

Session එක Reset වී නැවත Notebook එක Open කළ විට, Model එක ආයේ Internet එකෙන් Download කිරීමට සිදු නොවේ.

Google Drive එක Colab එකට Mount (Link) කිරීමට යන්නේ තත්පර 5-10ක් පමණි.

Drive එකේ ඇති Model file එක direct ComfyUI එකට සම්බන්ධ වන බැවින් තත්පර 30-40ක් ඇතුළත මුළු Environment එකම නැවත Active කරගත හැක.

Model එක permanent ලෙස තබා ගන්නා ආකාරය (Workflow)
Colab එකේ පළමු පියවර ලෙස පහත Code එක භාවිත කරන්න:

Python